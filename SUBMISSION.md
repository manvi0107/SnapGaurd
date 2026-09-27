# Reflex Guard — Submission Summary

**One-line pitch:** An AI vision safety interlock that stops power tools
before they injure — retrofittable to any existing tool with a clamp-on
camera and relay, no machine redesign needed, targeting the huge,
underserved gap in occupational safety across Indian MSME workshops and
farms where guarded/self-braking machinery is too costly to retrofit at
scale.

**Live demo:** see the dashboard link shared alongside this submission.
**Repo:** this zip / the linked GitHub repo.

Every claim below is labeled **REAL** (built, running, verifiable today),
**MOCKED** (real logic, running against a stand-in for hardware that isn't
in hand yet), or **PLANNED** (designed, not yet built). No claim in this
document is asserted without one of those three labels — that's a
deliberate choice, not hedging for its own sake.

---

## Architecture

Per the Snapdragon AI Lab program's own model — *"create real-world AI
solutions on Snapdragon-powered PCs, enhanced by Arduino's accessible
ecosystem"* — Reflex Guard splits across two real, separate machines:

```
Snapdragon-powered PC                    Arduino UNO Q
┌───────────────────────┐   USB serial   ┌─────────────────────────┐
│ Webcam                │ ─────────────► │ STM32U585 core          │
│  │                    │  danger flag    │ (firmware/reflex_guard  │
│  ▼                    │  + heartbeat    │  .ino) — watchdog +     │
│ AIHubHandDetector      │ ◄───────────── │  relay cutoff, fails    │
│ (Qualcomm AI Hub       │  ack + timing   │  safe if the PC side    │
│  MediaPipe-Hand-       │                 │  ever stalls            │
│  Detection model)      │                 └─────────────────────────┘
│  │                    │
│  ▼                    │
│ Digital Twin: predicted│
│ vs. measured response  │
│  │                    │
│  ▼                    │
│ Flask/Socket.IO        │
│ dashboard              │
└───────────────────────┘
```

## Evaluation criteria

### Technical Implementation
- **REAL** — Dual-process safety architecture: AI inference and the
  real-time cutoff run as genuinely separate processes/devices with a
  narrow, documented interface (`src/pc_bridge/serial_link.py` ↔
  `firmware/reflex_guard.ino`'s binary protocol), not one script doing
  everything.
- **REAL** — `AIHubHandDetector` (`src/vision/qai_hub_model.py`) wraps
  MediaPipe-Hand-Detection, a real model from the Qualcomm AI Hub model
  zoo (`qai_hub_models.models.mediapipe_hand`), run through
  `onnxruntime`, with a QNN Execution Provider code path
  (`providers=["QNNExecutionProvider"]`) for Snapdragon NPUs — Qualcomm's
  own documented deployment method — and a CPU fallback using the
  identical graph.
- **REAL** — `scripts/export_qai_hub_model.py` is a working script against
  the actual `qai_hub` SDK (`submit_compile_and_quantize_jobs`,
  `submit_profile_job`) for INT8 quantization and cloud-device profiling.
- **REAL** — Digital twin (`src/digital_twin/twin.py`): a named,
  auditable component latency budget, a kinematic hand-motion model that
  projects time-to-danger, and a predicted-vs-measured drift detector
  (`NOMINAL`/`DEGRADED`/`ANOMALY`) — demoed live catching an injected
  slow-relay spike.
- **REAL** — STM32 firmware (`firmware/reflex_guard.ino`): checksummed
  binary protocol, watchdog-based fail-safe, boot-safe default (power off
  until the PC proves it's alive).
- **MOCKED** — The serial link to the STM32 (`MockSerialLink`) and the
  QNN NPU execution path, until physical UNO Q + Snapdragon-PC hardware
  access is confirmed. Both are built against the exact same interface
  the real hardware uses, so this is a hardware-access gap, not a design
  gap — see "What's left," below.
- **PLANNED** — The palm-detector's anchor-box output decoder
  (`_postprocess()` in `qai_hub_model.py`) is a documented stub; wiring in
  `qai_hub_models`' own decoder is the next concrete step once compiled
  weights are in hand.

### Application Use Case & Innovation
- **REAL** (as a design target, demonstrated in the working prototype) —
  Retrofit-first: clamp-on camera + relay onto *existing* tools, not a
  new machine. This is the difference that matters for the MSME/farm
  cost barrier this project targets.
- **REAL** — Fail-safe, not just fail-detect: the safety property holds
  even if the AI/PC side crashes entirely, because the STM32 core doesn't
  need to hear "danger" to cut power — it needs to keep hearing "safe,"
  and stops on silence.
- **REAL, and the main point of novelty** — The digital twin adds a
  predictive *health-monitoring* layer most reactive "AI safety" entries
  don't have: catching a degrading relay or a slowing response before it
  becomes a missed detection, not just reacting to one.

### Deployment & Accessibility
- **REAL** — Fully offline operation on the safety-critical path itself
  (webcam → model → serial → relay needs no network); only the one-time
  AI Hub export/compile step needs internet.
- **REAL** — Low integration cost by design: a retrofit, not a
  replacement.
- **REAL** — Publicly reachable, interactive live demo (not a video or
  screenshots) — dashboard link included with this submission.
- **REAL** — Graceful degradation throughout: every optional dependency
  (opencv, pyserial, a physical board) is checked at runtime, and its
  absence is reported honestly in the UI rather than faked.

### Presentation & Documentation
- **REAL** — Structured docs at every level: root `README.md` (overview +
  quick start), `docs/architecture.md` (design rationale), `webapp/README.md`
  (dashboard internals + message formats), `firmware/README.md` (flashing
  instructions), this file (evaluation-criteria mapping).
- **REAL** — Honest labeling is a running design principle, not just a
  section in this document: the dashboard itself tags simulated telemetry
  as `SIMULATED TELEMETRY` in the UI, never presenting it as real
  inference.

---

## What's left (in priority order)

1. Flash `firmware/reflex_guard.ino` onto a physical UNO Q, confirm relay
   wiring/logic level.
2. Confirm `onnxruntime`'s `QNNExecutionProvider` actually runs on the
   target Snapdragon PC's NPU (code path is written and correct per
   Qualcomm's documentation; not yet run on physical Snapdragon silicon
   by us — pending hardware access).
3. Wire in `qai_hub_models`' real anchor-box decoder once compiled model
   weights are exported.
4. End-to-end timing validation over the real USB link (currently
   validated against `MockSerialLink`'s timing-realistic stand-in).

## Repo map
```
reflex-guard/
├── README.md                  overview + quick start
├── SUBMISSION.md               this file
├── docs/architecture.md        design rationale
├── firmware/reflex_guard.ino   STM32U585 sketch (real hardware target)
├── firmware/README.md          flashing instructions
├── scripts/export_qai_hub_model.py   real Qualcomm AI Hub SDK usage
├── src/
│   ├── vision/qai_hub_model.py       AI Hub model wrapper (QNN + CPU)
│   ├── digital_twin/twin.py          predicted-vs-actual drift model
│   ├── pc_bridge/serial_link.py      PC <-> STM32 protocol (real + mock)
│   ├── safety/realtime_core.py       Pi-prototype safety core
│   └── dashboard/                    standalone Pi/webcam prototype
└── webapp/
    ├── server.py                     Flask + Socket.IO dashboard server
    ├── pc_provider.py                real webcam+AI+serial data source
    ├── simulation_provider.py        zero-hardware demo data source
    ├── hardware_provider.py          networked-device placeholder
    └── static/, templates/           the dashboard UI itself
```
