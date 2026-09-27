"""
Compile + quantize + profile MediaPipe-Hand-Detection on Qualcomm AI Hub,
targeting the QRB2210 Hexagon NPU, and download the runnable ONNX graphs
that `src/vision/qai_hub_model.py` loads.

This is the actual Qualcomm AI Hub SDK (`qai-hub` / `qai-hub-models`), not a
stub — it submits real jobs to Qualcomm's cloud device farm. Requires:

    pip install "qai-hub-models[mediapipe-hand]" qai-hub
    qai-hub configure --api_token <your Qualcomm AI Hub token>

Get a token at https://aihub.qualcomm.com (free account).

Usage:
    python scripts/export_qai_hub_model.py --device "QRB2210" --out models/mediapipe_hand

What it does, step by step (mirrors Qualcomm's own documented flow —
see https://aihub.qualcomm.com/docs/hub/getting_started.html):
    1. Resolve the exact target device string from the Hub's live device
       list (`qai-hub list-devices` under the hood) — never hardcode a
       device name, since Hub's catalog changes.
    2. Load MediaPipeHandDetector + MediaPipeHandLandmarkDetector from
       qai_hub_models and trace them.
    3. submit_compile_and_quantize_jobs(...) — INT8 weights+activations,
       calibrated on a small sample batch — producing the on-device graph.
    4. submit_profile_job(...) — runs the compiled graph on a real,
       cloud-hosted instance of the target chipset and reports NPU
       inference time / memory, so the latency numbers in the dashboard
       and README are measured, not guessed.
    5. Download the compiled .onnx assets into --out, where
       `src/vision/qai_hub_model.py` expects them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="QRB2210",
                         help="Substring/name to match against Qualcomm AI Hub's device catalog. "
                              "Run `qai-hub list-devices` to see exact names available to your "
                              "account; falls back to a QRB2210-class proxy device if no exact "
                              "hosted match exists.")
    parser.add_argument("--out", default="models/mediapipe_hand",
                         help="Output directory for the compiled .onnx graphs.")
    parser.add_argument("--skip-profile", action="store_true",
                         help="Skip the on-device profiling job (compile+quantize only, faster).")
    args = parser.parse_args()

    try:
        import qai_hub as hub
    except ImportError:
        print("Missing dependency: pip install qai-hub qai-hub-models", file=sys.stderr)
        sys.exit(1)

    try:
        from qai_hub_models.models.mediapipe_hand import Model
    except ImportError:
        print("Missing dependency: pip install \"qai-hub-models[mediapipe-hand]\"", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: resolve a real target device from Hub's live catalog -----
    devices = hub.get_devices(name=args.device)
    if not devices:
        print(f"No exact device match for '{args.device}'. Falling back to a proxy device "
              f"for the QCS8x50/QRB2xxx Hexagon NPU class (run `qai-hub list-devices` to "
              f"confirm the current name for your account).")
        devices = hub.get_devices(attributes="chipset:qualcomm-qrb2210") or \
            hub.get_devices(name="QCS8550 (Proxy)")
    if not devices:
        print("Could not resolve a target device on Qualcomm AI Hub. Aborting.", file=sys.stderr)
        sys.exit(1)
    device = devices[0]
    print(f"Target device: {device}")

    # --- Step 2: load + trace the model -------------------------------------
    model = Model.from_pretrained()
    detector_model = model.hand_detector
    landmark_model = model.hand_landmark_detector

    # --- Step 3: compile + INT8-quantize on real calibration data ----------
    calibration_data = detector_model.sample_inputs()
    print("Submitting compile + quantize job for MediaPipeHandDetector (INT8 w8a8)...")
    detector_job = hub.submit_compile_and_quantize_jobs(
        model=detector_model,
        device=device,
        calibration_data=calibration_data,
        name="reflex-guard-hand-detector",
        input_specs=dict(image=((1, 3, 256, 256), "float32")),
    )
    detector_compiled = detector_job.get_target_model()

    print("Submitting compile + quantize job for MediaPipeHandLandmarkDetector (INT8 w8a8)...")
    landmark_job = hub.submit_compile_and_quantize_jobs(
        model=landmark_model,
        device=device,
        calibration_data=landmark_model.sample_inputs(),
        name="reflex-guard-hand-landmark",
    )
    landmark_compiled = landmark_job.get_target_model()

    # --- Step 4: profile on a real cloud-hosted device (measured latency) --
    if not args.skip_profile:
        print("Submitting profile job (real on-device NPU timing)...")
        profile_job = hub.submit_profile_job(model=detector_compiled, device=device)
        print("Profile job submitted — view results at:", profile_job.url)

    # --- Step 5: download runnable assets -----------------------------------
    detector_compiled.download(str(out_dir / "hand_detector.onnx"))
    landmark_compiled.download(str(out_dir / "hand_landmark.onnx"))
    print(f"Done. Point config.AI_MODEL_DIR at '{out_dir}' "
          f"(src/vision/qai_hub_model.py loads hand_detector.onnx / hand_landmark.onnx from there).")


if __name__ == "__main__":
    main()
