"""
Lightweight, dependency-light hand/object-proximity detector.

This uses OpenCV background subtraction + contour centroid tracking against
a configured danger-zone rectangle. It stands in for the quantized on-device
model (from Qualcomm AI Hub, run via QNN on the QRB2210's GPU/DSP) in the
real submission — same role in the pipeline (frame in, danger flag out),
lighter dependency footprint for fast prototyping on a Pi with no GPU
acceleration required.

Swap `Detector._analyze_frame` for a MediaPipe Hands / TFLite model call
later without touching anything else in the pipeline.
"""

import time
import math
import cv2
import numpy as np


class Detector:
    def __init__(self, config):
        self.cfg = config
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
        self._frame_count = 0

    def danger_zone_px(self, frame_shape):
        h, w = frame_shape[:2]
        z = self.cfg.DANGER_ZONE
        return (int(z["x1"] * w), int(z["y1"] * h), int(z["x2"] * w), int(z["y2"] * h))

    def process(self, frame):
        """Returns (annotated_frame, danger: bool, distance_px: float|None)."""
        self._frame_count += 1
        x1, y1, x2, y2 = self.danger_zone_px(frame.shape)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        if self._frame_count % self.cfg.DETECT_EVERY_N_FRAMES != 0:
            return frame, False, None

        danger, distance_px, centroid = self._analyze_frame(frame, (x1, y1, x2, y2))

        if centroid is not None:
            color = (0, 0, 255) if danger else (0, 220, 0)
            cv2.circle(frame, centroid, 8, color, -1)

        label = "DANGER" if danger else "SAFE"
        color = (0, 0, 255) if danger else (0, 220, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return frame, danger, distance_px

    def _analyze_frame(self, frame, zone_px):
        x1, y1, x2, y2 = zone_px
        fg_mask = self.bg_sub.apply(frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.cfg.MIN_CONTOUR_AREA and area > best_area:
                best = c
                best_area = area

        if best is None:
            return False, None, None

        m = cv2.moments(best)
        if m["m00"] == 0:
            return False, None, None
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

        # Distance from centroid to the danger-zone rectangle (0 if inside it)
        dx = max(x1 - cx, 0, cx - x2)
        dy = max(y1 - cy, 0, cy - y2)
        distance_px = math.hypot(dx, dy)
        danger = distance_px == 0

        return danger, distance_px, (cx, cy)


class CameraSource:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. "
                "Set MODE='simulate' in config.py to run without hardware."
            )

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def release(self):
        self.cap.release()


class SimulatedSource:
    """
    Generates synthetic frames with a moving 'hand' rectangle that sweeps
    in and out of the danger zone, for demos with no camera hardware.
    """

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.t0 = time.time()

    def read(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # dark background


        t = time.time() - self.t0
        # Hand sweeps left-to-right and back every 6 seconds, dipping into
        # the danger zone in the middle of each sweep.
        phase = (t % 6.0) / 6.0
        sweep = abs(math.sin(phase * math.pi))
        hand_x = int(self.w * (0.1 + 0.8 * sweep))
        hand_y = int(self.h * 0.5)

        cv2.rectangle(frame, (hand_x - 25, hand_y - 40), (hand_x + 25, hand_y + 40), (200, 200, 200), -1)
        cv2.putText(frame, "SIMULATED FEED", (10, self.h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return frame

    def release(self):
        pass


def build_detector(config):
    """
    Factory used by `src/dashboard/app.py`. Returns the real Qualcomm AI Hub
    detector (`AIHubHandDetector`) when `config.DETECTOR_BACKEND == "qai_hub"`
    (the default), or this module's dependency-light OpenCV heuristic
    otherwise. Both expose the same `.process(frame) -> (frame, danger,
    distance_px)` contract, so nothing downstream (RealtimeCore, the
    dashboard, the digital twin) needs to know which one is active.
    """
    backend = getattr(config, "DETECTOR_BACKEND", "opencv")
    if backend == "qai_hub":
        from .qai_hub_model import AIHubHandDetector
        return AIHubHandDetector(config)
    return Detector(config)
