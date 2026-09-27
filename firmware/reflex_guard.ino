/*
  Reflex Guard — STM32U585 safety firmware (Arduino UNO Q)
  =========================================================
  Runs on the UNO Q's STM32U585 real-time core. Talks to the Snapdragon PC
  over USB serial using a tiny, fixed-size binary protocol, and is the
  actual fail-safe authority: if it stops hearing from the PC, it treats
  that exactly the same as being told "danger" — cutting power is the
  default, staying on is the thing that has to be continuously earned.

  This intentionally mirrors `RealtimeCore` in src/safety/realtime_core.py
  (same contract: signal(danger) + watchdog), just implemented on real
  hardware instead of a Python thread.

  Wire protocol (PC -> STM32), 3 bytes per message:
    [0xA5] [CMD] [CHECKSUM]
    CMD: 0x01 = SAFE       (clear to run)
         0x02 = DANGER     (cut power now)
         0x00 = HEARTBEAT  (I'm alive, no state change)
    CHECKSUM = 0xA5 ^ CMD

  Wire protocol (STM32 -> PC), on every received message:
    [0x5A] [STATE] [MICROS_LOW] [MICROS_MID] [MICROS_HIGH] [MICROS_TOP]
    STATE: mirrors current relay state (0x01 = closed/running, 0x02 = open/tripped)
    MICROS_*: micros() timestamp of when this ack was sent, little-endian,
              used by the PC side to compute real round-trip latency for
              the digital twin (src/digital_twin/twin.py).

  Relay wiring: RELAY_PIN drives a solid-state relay / relay module,
  active-HIGH = power ON. Adjust to your relay module's actual logic level.
*/

#include <Arduino.h>

const uint8_t PIN_MAGIC_IN = 0xA5;
const uint8_t PIN_MAGIC_OUT = 0x5A;

const uint8_t CMD_HEARTBEAT = 0x00;
const uint8_t CMD_SAFE = 0x01;
const uint8_t CMD_DANGER = 0x02;

const int RELAY_PIN = 7;      // to the relay module's IN pin
const int STATUS_LED_PIN = 13; // onboard LED, mirrors relay state

// Watchdog: if the PC hasn't sent *anything* (heartbeat or otherwise)
// within this window, that's treated as the AI/Linux side stalling —
// fail-safe means we cut power, not that we keep the last known state.
const unsigned long WATCHDOG_TIMEOUT_MS = 150;

bool relayClosed = false; // false = OPEN/tripped (power off), matches boot-safe default
unsigned long lastMessageMs = 0;

void setRelay(bool closed) {
  relayClosed = closed;
  digitalWrite(RELAY_PIN, closed ? HIGH : LOW);
  digitalWrite(STATUS_LED_PIN, closed ? HIGH : LOW);
}

void sendAck() {
  uint8_t state = relayClosed ? 0x01 : 0x02;
  uint32_t t = micros();
  uint8_t buf[6] = {
    PIN_MAGIC_OUT, state,
    (uint8_t)(t & 0xFF), (uint8_t)((t >> 8) & 0xFF),
    (uint8_t)((t >> 16) & 0xFF), (uint8_t)((t >> 24) & 0xFF)
  };
  Serial.write(buf, sizeof(buf));
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  setRelay(false); // boot fail-safe: power OFF until the PC proves it's alive and SAFE
  Serial.begin(115200);
  lastMessageMs = millis();
}

void loop() {
  // --- Watchdog: no message in time -> fail-safe cutoff, independent of
  // whatever the PC/AI side is doing. This is the whole point of putting
  // this logic on the STM32 core rather than trusting the PC to behave. --
  if (millis() - lastMessageMs > WATCHDOG_TIMEOUT_MS) {
    if (relayClosed) setRelay(false);
  }

  // --- Read one 3-byte command, if available ------------------------------
  if (Serial.available() >= 3) {
    uint8_t magic = Serial.read();
    uint8_t cmd = Serial.read();
    uint8_t checksum = Serial.read();

    if (magic == PIN_MAGIC_IN && checksum == (uint8_t)(PIN_MAGIC_IN ^ cmd)) {
      lastMessageMs = millis();
      if (cmd == CMD_SAFE) {
        setRelay(true);
      } else if (cmd == CMD_DANGER) {
        setRelay(false);
      }
      // CMD_HEARTBEAT: no state change, just resets the watchdog above.
      sendAck();
    }
    // Malformed frame: dropped silently. The watchdog above is what
    // actually protects against a corrupted/dead link, not per-byte
    // error recovery here.
  }
}
