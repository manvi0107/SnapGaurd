# Reflex Guard

An AI vision safety interlock for unguarded power tools and machinery —
built for the [Snapdragon® AI Lab Build & Present Challenge](https://unstop.com/competitions/crp-snapdragon-ai-lab-build-present-challenge-qualcomm-1748893).

Per the program's own architecture (*"create real-world AI solutions on
Snapdragon-powered PCs, enhanced by Arduino's accessible ecosystem"*),
Reflex Guard splits across **two real, separate machines**: AI inference
runs on a **Snapdragon-powered PC**, and the safety-critical cutoff runs on
an **Arduino UNO Q**'s STM32U585 core, talking to each other over USB
serial. Camera watches the blade/cutting zone → **MediaPipe-Hand-Detection**
(a real model from the Qualcomm AI Hub model zoo, INT8-quantized, run
through ONNX Runtime — QNN on Snapdragon NPUs where available, CPU
elsewhere) detects a hand entering the danger zone on the PC → the danger
flag crosses USB serial → the STM32 core cuts power within single-digit
milliseconds, independent of whatever the PC/AI side is doing, and
fail-safe if that side ever stalls or the link drops.

A **digital twin** runs alongside the real system, predicting response
time from a named component latency budget and flagging drift if the real
hardware ever starts responding slower than its design spec — an
early-warning signal for a degrading relay or an overloaded link, not just
a post-hoc alarm.

See [`SUBMISSION.md`](SUBMISSION.md) for the full write-up (what's real,
what's simulated, and why), and [`docs/architecture.md`](docs/architecture.md)
for the technical design.

## Three parts to this repo

- **`webapp/`** — the judge-facing real-time dashboard, with three
  interchangeable data sources: a live **Simulation** (no hardware
  needed), **PC + UNO Q** (real webcam, real AI model, real USB serial
  link to real STM32 firmware — falls back to a timing-realistic mock
  link if no board is plugged in), and a **Live Hardware** placeholder
  for a future networked device. **Start here for a demo.** See
  [`webapp/README.md`](webapp/README.md).
- **`firmware/`** — the actual STM32U585 sketch (`reflex_guard.ino`) that
  runs on the Arduino UNO Q: a small serial protocol with a watchdog, so
  the relay fails safe if the PC side stops responding in time. See
  [`firmware/README.md`](firmware/README.md) for flashing instructions.
- **`src/`** — the Python/OpenCV edge pipeline (detector, digital twin,
  Qualcomm AI Hub model wrapper, PC↔serial bridge) shared by both the
  dashboard and a standalone Raspberry Pi prototype — useful for testing
  against real sensors before hardware is fully wired up.

## Quick start (dashboard, no hardware required)

```bash
git clone <your-repo-url>
cd reflex-guard/webapp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://localhost:5001`. To run **PC + UNO Q** mode with your own
webcam and (optionally) a real board, see the "PC LIVE mode" section in
[`webapp/README.md`](webapp/README.md). Full message formats and how to
wire in networked hardware later are also there.

## Quick start (edge pipeline, `src/`)

```bash
cd reflex-guard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.dashboard.app
```

Open `http://<device-ip>:5000` in a browser.

### No camera yet?
`src/config.py` defaults to `MODE = "simulate"`, which generates a synthetic
"hand" sweeping in and out of the danger zone — so the whole pipeline
(detection → safety core → dashboard) runs and demos end-to-end with zero
hardware. Once you have a USB webcam or Pi camera:

```bash
REFLEX_MODE=camera python -m src.dashboard.app
```

### Optional: real relay control on a Pi
Set `USE_GPIO = True` in `src/config.py` and wire a relay module to
`RELAY_PIN` (BCM numbering). Falls back to logging-only if `RPi.GPIO` isn't
installed, so the rest of the code runs unchanged on a laptop too.

## The AI model — Qualcomm AI Hub, on the NPU
`src/vision/qai_hub_model.py` wraps **MediaPipe-Hand-Detection**
(`qai_hub_models.models.mediapipe_hand`), Qualcomm's own AI Hub port of the
MediaPipe hand-landmark pipeline. It's a real, two-stage model (palm
detector + 21-point landmark model), not a placeholder:

- **`scripts/export_qai_hub_model.py`** submits real compile + INT8-quantize
  + profile jobs to Qualcomm AI Hub's cloud device farm, targeting the
  QRB2210's Hexagon NPU class, and downloads the runnable ONNX graphs.
- **`QNNBackend`** runs those graphs on-device through
  `onnxruntime.InferenceSession(..., providers=["QNNExecutionProvider"],
  provider_options=[{"backend_path": "libQnnHtp.so"}])` — Qualcomm's own
  documented path for running ONNX models on Dragonwing/QRB Linux NPUs.
- **`ONNXCPUBackend`** runs the identical graph on CPU for dev-machine /
  Raspberry Pi testing — same code path, same pre/post-processing, so
  what runs on a laptop today is what runs on the NPU on submission day.
- `config.DETECTOR_BACKEND` (`"qai_hub"` / `"opencv"`) and `config.AI_BACKEND`
  (`"npu"` / `"cpu"` / `"auto"`) select between these without touching any
  other file — see `src/vision/detector.py:build_detector()`.

`src/dashboard/app.py` exposes this at `/api/ai_status` (model, backend,
quantization, measured latency) so the dashboard never has to guess.

## Digital twin
`src/digital_twin/twin.py` is a small, auditable virtual model of the
hardware that runs alongside the real device:
- predicts end-to-end response time from a named component latency budget
  (camera capture → NPU inference → Linux↔MCU bridge → STM32 decision →
  relay actuation);
- mirrors hand position as a simple kinematic model and projects a
  time-to-danger before the hand reaches the zone;
- compares its prediction against real measured response times and flags
  `DEGRADED` / `ANOMALY` when they drift apart — useful as an early-warning
  signal for a slowing relay or an overloaded NPU, not just a demo toy.

See it live on the **Digital Twin** tab of the `webapp/` dashboard.

## How the pieces talk to each other
```
camera/simulated frames → AIHubHandDetector (QNN NPU / ONNX CPU) → danger flag
                                    │                                    │
                                    ▼                                    ▼
                             Digital Twin                      RealtimeCore (own thread, watchdog)
                        (predicted vs. actual)                            │
                                    │                                     ▼
                                    │                         relay trip (or logged event)
                                    │                                     │
                                    └──────────────► Flask dashboard ◄────┘
                                            (live feed, state, latency, event log, twin)
```

Full write-up in [`docs/architecture.md`](docs/architecture.md).

## Roadmap
- [x] Swap the OpenCV detector for a Qualcomm AI Hub quantized hand-detection
      model running via QNN
- [x] Digital twin: predicted-vs-actual response time, drift detection
- [x] Real STM32U585 firmware (`firmware/reflex_guard.ino`) implementing
      `RealtimeCore`'s watchdog/fail-safe contract over real USB serial
- [x] PC ↔ UNO Q live pipeline (`webapp/pc_provider.py`,
      `src/pc_bridge/serial_link.py`) — real webcam, real AI model, real
      serial link, with a timing-realistic mock fallback when no board is
      attached
- [ ] Flash `firmware/reflex_guard.ino` onto a physical UNO Q and confirm
      the relay wiring/logic level end-to-end (code is written and
      unit-tested against the mock link; physical bring-up is the
      remaining step, pending board access)
- [ ] Benchmark the compiled AI Hub model on a physical Snapdragon NPU via
      the real `QNNExecutionProvider` path (currently validated on
      CPU/ONNX Runtime + profiled on Qualcomm AI Hub's cloud device farm)
- [ ] Add audio (blade-sound signature) as a second detection modality
- [ ] Fleet dashboard aggregating near-miss events across a workshop

## License
MIT
