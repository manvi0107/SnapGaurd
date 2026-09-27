"""
RealtimeCore simulates the deterministic safety layer that, on the real
Snapdragon UNO Q hardware, lives on the STM32U585 real-time core rather than
on Linux. It is deliberately isolated:

  * it runs in its own thread with a tight poll loop
  * it only talks to the vision layer through `signal()` (one function call,
    no shared complex state)
  * it fails SAFE: if it stops hearing from the vision loop, it trips anyway

This isolation is the point of the architecture, not an implementation detail
— it's what would let this same module be reimplemented as STM32 firmware
without changing how the rest of the system talks to it.
"""

import threading
import time
from collections import deque

try:
    import RPi.GPIO as GPIO  # noqa: N814
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False


class RealtimeCore(threading.Thread):
    STATE_SAFE = "SAFE"
    STATE_DANGER = "DANGER"
    STATE_TRIPPED = "TRIPPED"

    def __init__(self, watchdog_timeout_s, poll_interval_s, use_gpio=False,
                 relay_pin=17, event_log_max=200):
        super().__init__(daemon=True)
        self._watchdog_timeout = watchdog_timeout_s
        self._poll_interval = poll_interval_s
        self._use_gpio = use_gpio and _HAS_GPIO
        self._relay_pin = relay_pin

        self._lock = threading.Lock()
        self._last_heartbeat = time.monotonic()
        self._state = self.STATE_SAFE
        self._trip_count = 0
        self._last_latency_ms = None
        self._events = deque(maxlen=event_log_max)
        self._stop_flag = threading.Event()

        if self._use_gpio:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._relay_pin, GPIO.OUT, initial=GPIO.HIGH)  # HIGH = relay closed / power on

    # ------------------------------------------------------------------ API
    def signal(self, danger: bool):
        """Called from the vision loop on every processed frame."""
        now = time.monotonic()
        with self._lock:
            self._last_heartbeat = now
            if danger and self._state != self.STATE_TRIPPED:
                self._trip(reason="hand detected in danger zone", detected_at=now)
            elif not danger and self._state == self.STATE_DANGER:
                self._state = self.STATE_SAFE

    def reset(self):
        with self._lock:
            if self._state == self.STATE_TRIPPED:
                self._log_event("manual reset")
            self._state = self.STATE_SAFE
            self._last_heartbeat = time.monotonic()
            if self._use_gpio:
                GPIO.output(self._relay_pin, GPIO.HIGH)

    def status(self):
        with self._lock:
            return {
                "state": self._state,
                "trip_count": self._trip_count,
                "last_latency_ms": self._last_latency_ms,
                "events": list(self._events),
            }

    def stop(self):
        self._stop_flag.set()

    # ------------------------------------------------------------- internal
    def run(self):
        while not self._stop_flag.is_set():
            with self._lock:
                idle_for = time.monotonic() - self._last_heartbeat
                if idle_for > self._watchdog_timeout and self._state != self.STATE_TRIPPED:
                    self._trip(reason=f"watchdog: no vision heartbeat for {idle_for:.2f}s",
                               detected_at=time.monotonic())
            time.sleep(self._poll_interval)

    def _trip(self, reason, detected_at):
        """Must be called with self._lock held."""
        actuate_start = time.monotonic()
        self._state = self.STATE_TRIPPED
        self._trip_count += 1
        if self._use_gpio:
            GPIO.output(self._relay_pin, GPIO.LOW)  # cut power
        actuate_done = time.monotonic()
        self._last_latency_ms = round((actuate_done - detected_at) * 1000, 2)
        self._log_event(f"TRIPPED — {reason} (cutoff in {self._last_latency_ms} ms)")

    def _log_event(self, message):
        self._events.append({"t": time.strftime("%H:%M:%S"), "message": message})
