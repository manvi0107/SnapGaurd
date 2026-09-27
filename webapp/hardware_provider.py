"""
HardwareProvider — the future real-device data source.

This is the seam where the actual Arduino UNO Q / STM32 system plugs in.
It implements the exact same contract as SimulationProvider
(`state`, `handle_command()`, `tick()`), so the frontend and the rest of
the server never need to change when you swap providers.

Right now there is no physical hardware, so this honestly reports
`hardwareConnected: False` and never fabricates live values — per the
"no hardcoded fake data" requirement, an unreachable device shows as
unreachable, not as a happy-looking dashboard.

To wire in a real device later:
  1. Point REFLEX_HW_ENDPOINT at your device's WebSocket/HTTP address.
  2. Implement `_try_connect()` to actually open that connection.
  3. Implement `_read_telemetry()` to parse whatever the device sends and
     map it into the same state shape SimulationProvider produces.
  4. Implement `handle_command()` to forward UI commands (start/stop,
     emergency stop, etc.) to the device instead of simulating them.
Everything else — the dashboard, charts, event log — needs no changes.
"""

import os
import socket
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.digital_twin import DigitalTwin  # noqa: E402

HW_ENDPOINT = os.environ.get("REFLEX_HW_ENDPOINT", "")  # e.g. "192.168.1.50:8765"


class HardwareProvider:
    def __init__(self, emit_state, emit_event, emit_log):
        self.emit_state = emit_state
        self.emit_event = emit_event
        self.emit_log = emit_log
        self._lock = threading.RLock()
        self.twin = DigitalTwin()
        self.state = {
            "mode": "live",
            "hardwareConnected": False,
            "system": {"state": "OFFLINE", "running": False},
            "ai": {"connected": False, "state": "DISCONNECTED", "confidence": 0.0, "latencyMs": 0.0},
            "safety": {"state": "DISCONNECTED", "heartbeat": None},
            "relay": {"state": "UNKNOWN"},
            "tool": {"power": "UNKNOWN"},
            "uptimeSeconds": 0,
            "faults": {"ai": False, "camera": False, "stm32": False, "relay": False, "comm": False},
            "aiModel": {
                "name": "MediaPipe-Hand-Detection",
                "source": "Qualcomm AI Hub (qai_hub_models.models.mediapipe_hand)",
                "quantization": "INT8 (w8a8)",
                "backend": "UNKNOWN",
                "computeUnit": "UNKNOWN",
                "npuUtilPct": None,
                "real": False,
            },
            "twin": self.twin.snapshot(),
        }
        self._try_connect_async()

    def _try_connect_async(self):
        threading.Thread(target=self._try_connect, daemon=True).start()

    def _try_connect(self):
        """Best-effort real connection attempt. Never fakes success."""
        if not HW_ENDPOINT:
            self.emit_log("LIVE HARDWARE: no REFLEX_HW_ENDPOINT configured")
            return
        host, _, port = HW_ENDPOINT.partition(":")
        try:
            with socket.create_connection((host, int(port or 80)), timeout=1.5):
                with self._lock:
                    self.state["hardwareConnected"] = True
                self.emit_log(f"LIVE HARDWARE: reached {HW_ENDPOINT}")
                # A real implementation would open a persistent WebSocket/serial
                # session here and stream telemetry into self.state.
        except OSError:
            self.emit_log(f"LIVE HARDWARE: could not reach {HW_ENDPOINT}")

    def tick(self):
        # No fabricated movement — state only changes when real telemetry
        # arrives from the device (not implemented yet).
        self.emit_state(self.snapshot())

    def snapshot(self):
        with self._lock:
            return dict(self.state, ai=dict(self.state["ai"]), safety=dict(self.state["safety"]),
                        relay=dict(self.state["relay"]), tool=dict(self.state["tool"]),
                        system=dict(self.state["system"]), faults=dict(self.state["faults"]))

    def handle_command(self, cmd):
        if not self.state["hardwareConnected"]:
            self.emit_log(f"LIVE HARDWARE: command '{cmd.get('action')}' ignored — no device connected")
            return
        # Forward to real device here once implemented.
        self.emit_log(f"LIVE HARDWARE: would forward command {cmd}")
