# Reflex Guard STM32 firmware

`reflex_guard.ino` runs on the Arduino UNO Q's STM32U585 real-time core.
It implements the fail-safe relay-cutoff logic described in
`src/safety/realtime_core.py` (this is that same logic, on real hardware
instead of a Python thread) and talks to the PC over USB serial using the
protocol documented at the top of the file, matching
`src/pc_bridge/serial_link.py` on the PC side.

## Flashing

1. Install the Arduino IDE (2.x) and the Arduino UNO Q board support
   package (Boards Manager → search "UNO Q").
2. Open `reflex_guard.ino`, select the correct board + port under
   Tools → Board / Port.
3. Check `RELAY_PIN` matches how you've wired your relay module, and that
   your relay module's active level matches `setRelay()`'s assumption
   (active-HIGH by default — flip the logic in `setRelay()` if yours is
   active-LOW).
4. Upload.

## Verifying it works standalone (before connecting the PC)

Open the IDE's Serial Monitor at 115200 baud. The board won't print
anything on its own (it's a binary protocol, not text logging) — that's
expected. To sanity-check it's alive, you can temporarily add
`Serial.println("alive")` inside `loop()` and confirm you see repeated
output, then remove it before real use (extra prints on the serial line
will corrupt the binary protocol frames the PC side expects).

## Boot-safe default

On power-up, before the PC has said anything, the relay is **OPEN**
(power off) — see `setRelay(false)` in `setup()`. The tool only powers on
once the PC sends an explicit `SAFE` command and the watchdog is
satisfied. This is intentional: an unpowered/unconnected safety system
should never default to "tool running."
