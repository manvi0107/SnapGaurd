"""
SimulationProvider — a virtual Reflex Guard device.

This is NOT a script of prerecorded values. It runs a live random-walk
telemetry loop (latency, confidence, heartbeat) and a real, time-stamped
state-machine sequence when a safety event is triggered — the "response
time" shown in the UI is a genuine elapsed-time measurement between real
timestamps recorded as the sequence runs, not a hardcoded number.

DataProvider contract (shared with HardwareProvider):
    .state                -> the current shared state dict
    .handle_command(cmd)  -> mutate state / kick off a sequence
    .tick()                -> called ~5x/second by the background loop
Both providers write into the same `state` shape and go through the same
`emit_state()` / `emit_event()` callbacks, so the frontend never needs to
know which one is active.
"""

import os
import random
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.digital_twin import DigitalTwin  # noqa: E402

STATES_AI = ["IDLE", "MONITORING", "HAND_DETECTED", "DANGER"]

AI_MODEL_INFO = {
    "name": "MediaPipe-Hand-Detection",
    "source": "Qualcomm AI Hub (qai_hub_models.models.mediapipe_hand)",
    "quantization": "INT8 (w8a8)",
}


def now_iso():
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"


class SimulationProvider:
    def __init__(self, emit_state, emit_event, emit_log):
        self.emit_state = emit_state
        self.emit_event = emit_event
        self.emit_log = emit_log
        self._lock = threading.RLock()
        self._start_time = time.monotonic()
        self._sequence_running = False
        self.twin = DigitalTwin()

        self.state = {
            "mode": "simulation",
            "hardwareConnected": True,  # simulator is always "connected" to itself
            "system": {"state": "STOPPED", "running": False},
            "ai": {"connected": True, "state": "IDLE", "confidence": 0.0, "latencyMs": 0.0},
            "safety": {"state": "DISARMED", "heartbeat": None},
            "relay": {"state": "CLOSED"},
            "tool": {"power": "OFF"},
            "uptimeSeconds": 0,
            "faults": {"ai": False, "camera": False, "stm32": False, "relay": False, "comm": False},
            "aiModel": {
                **AI_MODEL_INFO,
                "backend": "QNN (Hexagon NPU)",
                "computeUnit": "NPU (Hexagon)",
                "npuUtilPct": 0.0,
                "real": False,  # this is the simulator, not real inference — never claim otherwise
            },
            "twin": self.twin.snapshot(),
        }

    # ------------------------------------------------------------- ticking
    def tick(self):
        with self._lock:
            if self.state["system"]["running"]:
                self.state["uptimeSeconds"] = round(time.monotonic() - self._start_time, 1)
                if not self._sequence_running:
                    base_conf = 0.0 if self.state["ai"]["state"] == "IDLE" else 0.6
                    self.state["ai"]["confidence"] = round(
                        max(0, min(99.9, base_conf * 100 + random.uniform(-3, 3))), 1
                    )
                    self.state["ai"]["latencyMs"] = round(max(1.0, random.gauss(9, 2)), 2)
                self.state["safety"]["heartbeat"] = now_iso()
                base_util = 8.0 if self.state["ai"]["state"] == "IDLE" else 34.0
                self.state["aiModel"]["npuUtilPct"] = round(
                    max(2.0, min(97.0, base_util + random.uniform(-4, 6))), 1)
            if self.state["faults"]["ai"]:
                self.state["ai"]["connected"] = False
            if self.state["faults"]["stm32"]:
                self.state["safety"]["state"] = "DISCONNECTED"
            self.state["twin"] = self.twin.snapshot()
        self.emit_state(self.snapshot())

    def snapshot(self):
        with self._lock:
            return dict(self.state, ai=dict(self.state["ai"]), safety=dict(self.state["safety"]),
                        relay=dict(self.state["relay"]), tool=dict(self.state["tool"]),
                        system=dict(self.state["system"]), faults=dict(self.state["faults"]))

    # ------------------------------------------------------------ commands
    def handle_command(self, cmd):
        action = cmd.get("action")
        with self._lock:
            if action == "start_system":
                self._start_time = time.monotonic()
                self.state["system"] = {"state": "SAFE", "running": True}
                self.state["ai"]["state"] = "MONITORING"
                self.state["safety"]["state"] = "ARMED"
                self.state["relay"]["state"] = "CLOSED"
                self.state["tool"]["power"] = "ON"
                self._log_event("SYSTEM ARMED", "SYSTEM", None, "—", "SAFE")
                self.emit_log("SYSTEM ARMED")

            elif action == "stop_system":
                self.state["system"] = {"state": "STOPPED", "running": False}
                self.state["ai"]["state"] = "IDLE"
                self.state["safety"]["state"] = "DISARMED"
                self.state["tool"]["power"] = "OFF"
                self.emit_log("SYSTEM STOPPED")

            elif action == "restart_system":
                self.handle_command({"action": "stop_system"})
                time.sleep(0.05)
                self.handle_command({"action": "start_system"})
                return

            elif action == "set_tool":
                self.state["tool"]["power"] = "ON" if cmd.get("value") == "on" else "OFF"
                self._log_event(f"TOOL POWER {self.state['tool']['power']}", "USER", None, "—", self.state["system"]["state"])

            elif action == "set_mode":
                self.state["mode"] = cmd.get("value", "simulation")

            elif action == "inject_fault":
                target = cmd.get("target")
                if target in self.state["faults"]:
                    self.state["faults"][target] = True
                    self.emit_log(f"FAULT INJECTED: {target.upper()} DISCONNECTED")
                    self._log_event(f"FAULT: {target.upper()} disconnected", "FAULT", None, "—", "FAULT")

            elif action == "clear_fault":
                target = cmd.get("target")
                if target in self.state["faults"]:
                    self.state["faults"][target] = False
                    if target == "ai":
                        self.state["ai"]["connected"] = True
                    if target == "stm32":
                        self.state["safety"]["state"] = "ARMED" if self.state["system"]["running"] else "DISARMED"
                    self.emit_log(f"FAULT CLEARED: {target.upper()}")

            elif action == "emergency_stop":
                self._log_event("EMERGENCY STOP", "USER", None, "IMMEDIATE", "FAULT")
                self.state["system"]["state"] = "FAULT"
                self.state["relay"]["state"] = "OPEN"
                self.state["tool"]["power"] = "OFF"
                self.emit_log("EMERGENCY STOP TRIGGERED")

            elif action == "reset_safety" or action == "clear_safety_event":
                self.state["system"]["state"] = "SAFE" if self.state["system"]["running"] else "STOPPED"
                self.state["ai"]["state"] = "MONITORING" if self.state["system"]["running"] else "IDLE"
                self.state["relay"]["state"] = "CLOSED"
                self.state["tool"]["power"] = "ON" if self.state["system"]["running"] else "OFF"
                self._log_event("SAFETY SYSTEM RESET", "USER", None, "—", "SAFE")
                self.emit_log("SAFETY SYSTEM RESET")

            elif action in ("set_hand", "trigger_safety_event"):
                value = cmd.get("value", "enter_danger")
                if value in ("enter_danger",) or action == "trigger_safety_event":
                    threading.Thread(target=self._run_safety_sequence, daemon=True).start()
                elif value == "detected":
                    self.state["ai"]["state"] = "HAND_DETECTED"
                    self._log_event("HAND DETECTED", "CAMERA", round(random.uniform(85, 97), 1), "MONITORING", "SAFE")
                elif value in ("none", "exit_danger"):
                    self.state["ai"]["state"] = "MONITORING" if self.state["system"]["running"] else "IDLE"

        self.emit_state(self.snapshot())

    # ----------------------------------------------------- event sequence
    def _run_safety_sequence(self):
        if self._sequence_running:
            return
        self._sequence_running = True
        t_start = time.perf_counter()

        # Timing ranges are calibrated to sit near the digital twin's
        # component latency budget (src/digital_twin/twin.py LATENCY_BUDGET_MS,
        # nominal total ~11ms) rather than picked arbitrarily, so the twin's
        # predicted-vs-actual comparison on the Digital Twin tab has a
        # realistic amount of natural jitter around NOMINAL rather than
        # always reading as an anomaly.
        steps = [
            ("HAND DETECTED", "CAMERA", lambda: round(random.uniform(88, 97), 1), 0.002, 0.004, "MONITORING"),
            ("DANGER ZONE INTRUSION", "AI VISION", lambda: round(random.uniform(90, 99), 1), 0.002, 0.004, "DANGER"),
            ("SAFETY EVENT GENERATED", "AI VISION", lambda: None, 0.0015, 0.003, "DANGER"),
            ("STM32 RECEIVED EVENT", "SAFETY CORE", lambda: None, 0.001, 0.002, "DANGER"),
            ("RELAY OPENED", "SAFETY CORE", lambda: None, 0.001, 0.0018, "FAULT"),
            ("TOOL POWER OFF", "SAFETY CORE", lambda: None, 0.0008, 0.0015, "FAULT"),
        ]

        with self._lock:
            self.state["ai"]["state"] = "DANGER"
        # Simulated hand closing in on the danger zone, in mm, feeding the
        # digital twin's kinematic model so it can project a time-to-danger
        # before the "DANGER ZONE INTRUSION" step actually lands.
        for d in (180, 120, 70, 30):
            self.twin.update_hand_position(d)
            time.sleep(0.01)

        for label, source, conf_fn, lo, hi, sys_state in steps:
            time.sleep(random.uniform(lo, hi))
            with self._lock:
                elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
                if label == "RELAY OPENED":
                    self.state["relay"]["state"] = "OPEN"
                if label == "TOOL POWER OFF":
                    self.state["tool"]["power"] = "OFF"
                    self.twin.update_hand_position(0)
                    # The `elapsed_ms` above is deliberately paced (via the
                    # per-step sleep ranges) so a human can watch each stage
                    # land in the event log — it's a UI-pacing timer, not a
                    # hardware timing measurement. What the digital twin
                    # compares against its component latency budget instead
                    # is a modeled hardware-class response: predicted
                    # nominal + realistic jitter, with a small chance of an
                    # injected slow-relay spike, so the ANOMALY path in
                    # "Twin health" is something you can actually trigger
                    # and see caught, not a permanently-red number.
                    modeled_actual_ms = self.twin.state.predicted_response_ms + max(0.2, random.gauss(1.2, 2.2))
                    if random.random() < 0.12:
                        modeled_actual_ms += random.uniform(12, 30)  # simulated relay wear/spike
                    self.twin.observe_actual_response(round(modeled_actual_ms, 2))
                if label == "STM32 RECEIVED EVENT":
                    self.state["safety"]["state"] = "TRIPPED"
                self.state["system"]["state"] = sys_state
                self.state["twin"] = self.twin.snapshot()
                self._log_event(label, source, conf_fn(), f"{elapsed_ms} ms", sys_state)
            self.emit_state(self.snapshot())
            self.emit_log(f"{label} (t+{elapsed_ms}ms)")

        self._sequence_running = False

    def _log_event(self, event, source, confidence, action, status):
        entry = {
            "time": now_iso(),
            "event": event,
            "source": source,
            "confidence": confidence,
            "action": action,
            "status": status,
        }
        self.emit_event(entry)
