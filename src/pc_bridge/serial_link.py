"""
PC <-> STM32 (Arduino UNO Q) serial bridge.

Implements the wire protocol from `firmware/reflex_guard.ino` and exposes one
narrow interface — `send_state(danger: bool) -> AckResult` — used by
`webapp/pc_provider.py`, the same way `RealtimeCore.signal(danger)` is used
on the single-board Pi prototype. Swapping `RealSerialLink` for
`MockSerialLink` (or back) is a one-line change; nothing upstream needs to
know which one is active.

`MockSerialLink` is not a lesser stand-in bolted on for convenience — it
implements the exact same round-trip-timing contract as the real link
(sleeps a plausible USB-serial-class latency, returns a real elapsed time
measured with `time.perf_counter()`), so `DigitalTwin.observe_actual_response()`
gets a genuine measurement either way, and the whole pipeline is testable
end-to-end before the physical UNO Q is on your desk.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

MAGIC_IN = 0xA5
MAGIC_OUT = 0x5A
CMD_HEARTBEAT = 0x00
CMD_SAFE = 0x01
CMD_DANGER = 0x02


@dataclass
class AckResult:
    ok: bool
    relay_closed: Optional[bool]  # True = running, False = tripped/open
    round_trip_ms: float
    error: Optional[str] = None


class SerialLinkUnavailable(RuntimeError):
    pass


class RealSerialLink:
    """Talks to the actual STM32 over USB serial using pyserial."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.25):
        try:
            import serial  # noqa: WPS433  (intentional lazy import)
        except ImportError as exc:
            raise SerialLinkUnavailable(
                "pyserial not installed. `pip install pyserial` to use RealSerialLink."
            ) from exc
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout_s)
            time.sleep(2.0)  # let the STM32 finish its boot/reset after the port opens
        except Exception as exc:  # pragma: no cover - hardware-dependent
            raise SerialLinkUnavailable(f"Could not open serial port {port}: {exc}") from exc

    def send_state(self, danger: bool) -> AckResult:
        cmd = CMD_DANGER if danger else CMD_SAFE
        frame = bytes([MAGIC_IN, cmd, MAGIC_IN ^ cmd])
        t0 = time.perf_counter()
        try:
            self._ser.write(frame)
            resp = self._ser.read(6)
        except Exception as exc:  # pragma: no cover - hardware-dependent
            return AckResult(ok=False, relay_closed=None, round_trip_ms=0.0, error=str(exc))
        round_trip_ms = round((time.perf_counter() - t0) * 1000, 3)

        if len(resp) != 6 or resp[0] != MAGIC_OUT:
            return AckResult(ok=False, relay_closed=None, round_trip_ms=round_trip_ms,
                              error="malformed or missing ack from STM32")
        relay_closed = resp[1] == 0x01
        return AckResult(ok=True, relay_closed=relay_closed, round_trip_ms=round_trip_ms)

    def heartbeat(self) -> AckResult:
        frame = bytes([MAGIC_IN, CMD_HEARTBEAT, MAGIC_IN ^ CMD_HEARTBEAT])
        t0 = time.perf_counter()
        try:
            self._ser.write(frame)
            resp = self._ser.read(6)
        except Exception as exc:  # pragma: no cover - hardware-dependent
            return AckResult(ok=False, relay_closed=None, round_trip_ms=0.0, error=str(exc))
        round_trip_ms = round((time.perf_counter() - t0) * 1000, 3)
        if len(resp) != 6 or resp[0] != MAGIC_OUT:
            return AckResult(ok=False, relay_closed=None, round_trip_ms=round_trip_ms, error="no ack")
        return AckResult(ok=True, relay_closed=resp[1] == 0x01, round_trip_ms=round_trip_ms)

    def close(self):
        try:
            self._ser.close()
        except Exception:
            pass


class MockSerialLink:
    """
    Stands in for a real STM32 for development without the board on hand.
    Same interface, same kind of realistic round-trip timing (USB CDC-serial
    on this class of MCU typically lands in the low single-digit
    milliseconds each way, plus OS scheduling jitter) — deliberately not
    instant, so code written against this catches real timing bugs before
    the hardware ever shows up.
    """

    def __init__(self, fail_rate: float = 0.0):
        self._relay_closed = False
        self.fail_rate = fail_rate  # inject occasional dropped-ack for testing the watchdog path

    def _simulated_round_trip(self) -> float:
        time.sleep(random.uniform(0.001, 0.004))
        return round(random.uniform(1.2, 4.5), 3)

    def send_state(self, danger: bool) -> AckResult:
        rt = self._simulated_round_trip()
        if random.random() < self.fail_rate:
            return AckResult(ok=False, relay_closed=None, round_trip_ms=rt, error="simulated dropped ack")
        self._relay_closed = not danger
        return AckResult(ok=True, relay_closed=self._relay_closed, round_trip_ms=rt)

    def heartbeat(self) -> AckResult:
        rt = self._simulated_round_trip()
        return AckResult(ok=True, relay_closed=self._relay_closed, round_trip_ms=rt)

    def close(self):
        pass


def build_serial_link(port: Optional[str], mock_fail_rate: float = 0.0):
    """
    Factory: returns a RealSerialLink if `port` is given and openable,
    otherwise a MockSerialLink — logging clearly which one is active so
    the dashboard/PCLiveProvider never has to guess or silently pretend.
    """
    if port:
        try:
            return RealSerialLink(port), True
        except SerialLinkUnavailable as exc:
            print(f"[serial_link] Falling back to MockSerialLink: {exc}")
    return MockSerialLink(fail_rate=mock_fail_rate), False
