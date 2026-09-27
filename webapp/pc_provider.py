"""
PCLiveProvider — the Snapdragon-PC + Arduino UNO Q data source.

This is the third leg of the "Snapdragon AI Lab" architecture: real AI
inference (`AIHubHandDetector`, via Qualcomm AI Hub's MediaPipe-Hand-
Detection, on whatever NPU/GPU/CPU backend onnxruntime picks on this PC)
running against this machine's own webcam, talking over real USB serial
to the actual STM32U585 core on an Arduino UNO Q (`firmware/reflex_guard.ino`)
for the fail-safe cutoff.

Implements the exact same contract as `SimulationProvider` / `HardwareProvider`
(`state`, `handle_command()`, `tick()`, `snapshot()`), so nothing in
`server.py` or the frontend needs to know a third provider exists beyond
one dict entry and one button — see the "Adding a data source" note in
webapp/README.md.

Honesty note: if no webcam or no serial port/UNO Q is available, this
degrades to `MockSerialLink` (see src/pc_bridge/serial_link.py) and reports
`hardwareConnected: False` / `aiModel.real: False` rather than faking a
connection — same rule the other two providers follow.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config as rg_config  # noqa: E402
from src.vision.detector import build_detector  # noqa: E402
from src.digital_twin import DigitalTwin  # noqa: E402
from src.pc_bridge import build_serial_link  # noqa: E402

SERIAL_PORT = os.environ.get("REFLEX_SERIAL_PORT", "")  # e.g. "COM5" or "/dev/ttyACM0"
CAMERA_INDEX = int(os.environ.get("REFLEX_CAMERA_INDEX", "0"))
HEARTBEAT_INTERVAL_S = 0.05  # keeps the STM32 watchdog fed even with no danger transitions


class PCLiveProvider:
    def __init__(self, emit_state, emit_event, emit_log):
        self.emit_state = emit_state
        self.emit_event = emit_event
        self.emit_log = emit_log
        self._lock = threading.RLock()
        self._start_time = time.monotonic()
        self._running = False
        self._thread = None
        self._latest_jpeg = None

        self.detector = build_detector(rg_config)
        self.twin = DigitalTwin()
        self.link, self._real_serial = build_serial_link(SERIAL_PORT or None)

        self.state = {
            "mode": "pc",
            "hardwareConnected": self._real_serial,
            "system": {"state": "STOPPED", "running": False},
            "ai": {"connected": True, "state": "IDLE", "confidence": 0.0, "latencyMs": 0.0},
            "safety": {"state": "DISARMED" if self._real_serial else "DISCONNECTED", "heartbeat": None},
            "relay": {"state": "UNKNOWN"},
            "tool": {"power": "UNKNOWN"},
            "uptimeSeconds": 0,
            "faults": {"ai": False, "camera": False, "stm32": not self._real_serial, "relay": False, "comm": False},
            "aiModel": {
                "name": getattr(self.detector, "model_name", "MediaPipe-Hand-Detection"),
                "source": "Qualcomm AI Hub (qai_hub_models.models.mediapipe_hand)",
                "quantization": getattr(self.detector, "quantization", "n/a"),
                "backend": self.detector.status()["backend"] if hasattr(self.detector, "status") else "opencv heuristic",
                "computeUnit": self.detector.status()["compute_unit"] if hasattr(self.detector, "status") else "CPU",
                "npuUtilPct": None,
                "real": getattr(self.detector, "real_model", False),
            },
            "twin": self.twin.snapshot(),
            "serial": {"connected": self._real_serial, "port": SERIAL_PORT or "(mock link)"},
        }
        self.emit_log(
            f"PC LIVE provider ready — serial: {'REAL (' + SERIAL_PORT + ')' if self._real_serial else 'MOCK (no UNO Q detected)'}"
        )

    # ------------------------------------------------------------- lifecycle
    def _capture_loop(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            with self._lock:
                self.state["faults"]["camera"] = True
                self.state["ai"]["connected"] = False
            self.emit_log(f"ERROR: could not open camera index {CAMERA_INDEX}")
            return

        last_heartbeat = 0.0
        prev_danger = False
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                annotated, danger, distance_px = self.detector.process(frame)
                infer_ms = round((time.perf_counter() - t0) * 1000, 3)

                now = time.monotonic()
                ack = None
                if danger != prev_danger:
                    ack = self.link.send_state(danger)
                elif now - last_heartbeat > HEARTBEAT_INTERVAL_S:
                    ack = self.link.heartbeat()
                    last_heartbeat = now

                ok_jpeg, buf = cv2.imencode(".jpg", annotated)
                with self._lock:
                    if ok_jpeg:
                        self._latest_jpeg = buf.tobytes()
                    self.state["ai"]["state"] = "DANGER" if danger else ("MONITORING" if self.state["system"]["running"] else "IDLE")
                    self.state["ai"]["latencyMs"] = infer_ms
                    if distance_px is not None:
                        self.twin.update_hand_position(distance_px)
                    if ack is not None:
                        if ack.ok:
                            self.state["hardwareConnected"] = True
                            self.state["safety"]["state"] = "TRIPPED" if danger else "ARMED"
                            self.state["safety"]["heartbeat"] = now
                            self.state["relay"]["state"] = "CLOSED" if ack.relay_closed else "OPEN"
                            self.state["tool"]["power"] = "ON" if ack.relay_closed else "OFF"
                            self.state["faults"]["stm32"] = False
                            if danger != prev_danger:
                                self.twin.observe_actual_response(ack.round_trip_ms)
                                self.emit_event({
                                    "time": time.strftime("%H:%M:%S"),
                                    "event": "DANGER ZONE INTRUSION" if danger else "ZONE CLEARED",
                                    "source": "PC LIVE (real serial)" if self._real_serial else "PC LIVE (mock serial)",
                                    "confidence": None,
                                    "action": f"{ack.round_trip_ms} ms",
                                    "status": "DANGER" if danger else "SAFE",
                                })
                        else:
                            self.state["faults"]["stm32"] = True
                            self.emit_log(f"WARNING: no ack from STM32 ({ack.error})")
                    self.state["twin"] = self.twin.snapshot()
                    self.state["uptimeSeconds"] = round(time.monotonic() - self._start_time)
                prev_danger = danger
                self.emit_state(self.snapshot())
        finally:
            cap.release()

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    # --------------------------------------------------------------- contract
    def handle_command(self, cmd: dict):
        action = cmd.get("action")
        with self._lock:
            if action == "start_system" and not self._running:
                self._running = True
                self.state["system"] = {"state": "MONITORING", "running": True}
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                self.emit_log("PC LIVE: camera + detector + serial loop started")
            elif action == "stop_system":
                self._running = False
                self.state["system"] = {"state": "STOPPED", "running": False}
                self.emit_log("PC LIVE: stopped")
            elif action == "reset_safety":
                self.link.send_state(False)
                self.state["safety"]["state"] = "ARMED"

    def tick(self):
        # The capture thread drives its own state updates at camera frame
        # rate; this periodic tick only needs to keep uptime/heartbeat
        # fresh when the system is stopped (no capture thread running).
        with self._lock:
            if not self.state["system"]["running"]:
                self.state["uptimeSeconds"] = round(time.monotonic() - self._start_time)
        self.emit_state(self.snapshot())

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)
