# Architecture

## Target hardware (submission)
- **Linux side — Qualcomm Dragonwing QRB2210:** camera capture + AI
  inference. **MediaPipe-Hand-Detection**, sourced from the Qualcomm AI Hub
  model zoo (`qai_hub_models.models.mediapipe_hand`), INT8-quantized via
  `qai_hub.submit_compile_and_quantize_jobs`, runs through ONNX Runtime's
  `QNNExecutionProvider` on the Hexagon NPU (`src/vision/qai_hub_model.py`).
- **Real-time side — STM32U585 MCU:** deterministic cutoff logic. Receives a
  danger flag over the UNO Q's built-in Linux↔MCU bridge and trips a relay
  within single-digit milliseconds, independent of whatever the Linux side
  is doing. Includes a watchdog: if it stops hearing from the Linux side, it
  trips anyway (fail-safe).
- **Digital twin — runs on either side, or off-device:** a small predictive
  model (`src/digital_twin/twin.py`) that estimates response time from a
  named component latency budget, projects time-to-danger from hand motion,
  and flags drift between predicted and measured response — a continuous
  health check on the safety chain itself, not just on the workpiece.

## This prototype (Raspberry Pi, no Arduino/STM32 available)
Both roles run on one Pi, but as **separate threads with a narrow interface**
— `RealtimeCore.signal(danger: bool)` — specifically so the safety thread
(`src/safety/realtime_core.py`) is a drop-in replacement target for real
STM32 firmware later, without touching the vision or dashboard code. The
AI detector is behind the same kind of narrow interface
(`build_detector(config)` in `src/vision/detector.py`), so swapping the
placeholder OpenCV heuristic for the real Qualcomm AI Hub model — or
swapping the model's own NPU/CPU backend — is a config change, not a
rewrite.

```
 Camera / simulated frames
          │
          ▼
  AIHubHandDetector                 ┌─────────────┐
  (QNN NPU / ONNX CPU)  ──────────► │ Digital Twin │ ◄── real response times
          │                         └──────┬───────┘
          │ danger flag                    │ predicted vs. actual, drift
          ▼                                │
  RealtimeCore (own thread) ────────────────┘
          │
          ▼
  relay trip / GPIO
          │
          └──────────────► Flask dashboard ◄───────┘
                     (video feed + live status + event log + twin panel)
```

## Why the split matters
A single Linux-side "if danger: cut power" check is at the mercy of Python's
GIL, OS scheduling, and whatever else is running. Isolating the cutoff
decision in its own tight loop — with a watchdog that fails safe if the
vision side stalls — is what makes this a *safety* system rather than a
monitoring dashboard with an alarm bolted on. That's true on the Pi
prototype, and it's the same reason the real submission needs the STM32
core rather than doing everything in Linux. The digital twin extends this
principle one step further: instead of only reacting when the cutoff is
too slow, it continuously predicts what "too slow" should look like from
the component specs and flags the moment reality starts drifting from
that prediction — catching a degrading relay or an overloaded NPU before
it becomes a missed detection.

## Why a digital twin, specifically (not just a dashboard chart)
Three concrete things it does that a raw latency chart doesn't:
1. **Names its assumptions.** `LATENCY_BUDGET_MS` in `twin.py` is a
   per-stage budget with a stated source (AI Hub profiling figures,
   datasheet-class MCU/relay timing) — auditable, not a fitted curve.
2. **Predicts before the event, not just measures after.** The kinematic
   model (`update_hand_position`) projects a time-to-danger from closing
   velocity, which is what lets the dashboard show a countdown instead of
   only a post-hoc "it worked" / "it didn't."
3. **Turns a threshold breach into a maintenance signal.** `NOMINAL` /
   `DEGRADED` / `ANOMALY` on drift, not just pass/fail on the safety
   outcome itself — a relay that's still fast enough to be safe *today*
   but is trending slower is exactly the kind of thing a retrofit safety
   product for MSME workshops needs to surface before it fails.

## Swapping in real hardware later
- Replace `CameraSource` with the QRB2210 camera pipeline.
- Run `scripts/export_qai_hub_model.py` against your Qualcomm AI Hub
  account to compile/quantize/profile the model for your exact target
  device, then point `config.AI_MODEL_DIR` at the output — no code change.
- Set `config.AI_BACKEND = "npu"` once `onnxruntime-qnn` and the QAIRT
  `libQnnHtp.so` are present on-device.
- Replace `RealtimeCore` with STM32U585 firmware that exposes the same
  `signal(danger)` / watchdog contract over the Linux↔MCU bridge, and feed
  its real trip timestamps into `DigitalTwin.observe_actual_response()`.
- Dashboard code (Flask app + templates) is hardware-agnostic and can be
  reused as-is, pointed at the real device's status endpoint.
