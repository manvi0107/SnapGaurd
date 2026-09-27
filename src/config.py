"""
Central configuration for Reflex Guard.

On the real Snapdragon UNO Q submission, the vision pipeline runs on the
Dragonwing QRB2210 (Linux side) and the safety cutoff runs on the STM32U585
real-time core. This prototype runs both on a Raspberry Pi for development —
`src/safety/realtime_core.py` is written as an isolated thread with its own
timing loop specifically so that swapping it for real STM32 firmware later
is a drop-in replacement, not a redesign.
"""

# --- Frame source -----------------------------------------------------------
# "camera"   -> use a real webcam / Pi camera via OpenCV
# "simulate" -> synthetic frames, no camera hardware required
MODE = "simulate"          # override with env var REFLEX_MODE=camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Danger zone -------------------------------------------------------------
# Rectangle in normalized coordinates (0..1) representing the blade / cut zone.
DANGER_ZONE = {"x1": 0.35, "y1": 0.30, "x2": 0.65, "y2": 0.70}

# --- Detection ---------------------------------------------------------------
MIN_CONTOUR_AREA = 1200     # ignore small noise blobs
DETECT_EVERY_N_FRAMES = 1   # set >1 to reduce CPU load on slower Pi models

# --- AI model (Qualcomm AI Hub) -----------------------------------------------
# "qai_hub"  -> src/vision/qai_hub_model.py: real MediaPipe-Hand-Detection
#               model from Qualcomm AI Hub, run via ONNX Runtime.
# "opencv"   -> src/vision/detector.py: dependency-light background
#               subtraction heuristic (no model weights required).
DETECTOR_BACKEND = "qai_hub"       # override with env var REFLEX_DETECTOR=opencv
AI_BACKEND = "auto"                # "npu" | "cpu" | "auto" -- see qai_hub_model.py
AI_MODEL_DIR = "models/mediapipe_hand"   # populated by scripts/export_qai_hub_model.py
QNN_BACKEND_LIB = "libQnnHtp.so"   # QAIRT SDK path on the QRB2210 target device

# --- Safety core --------------------------------------------------------------
WATCHDOG_TIMEOUT_S = 0.5    # if no heartbeat from vision loop in this long -> fail-safe trip
POLL_INTERVAL_S = 0.005     # real-time core loop tick

# --- Actuation ----------------------------------------------------------------
# If RPi.GPIO is available and USE_GPIO is True, pin RELAY_PIN is driven LOW
# on trip (active-low relay module). Otherwise trips are logged only.
USE_GPIO = False
RELAY_PIN = 17

# --- Dashboard ------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 5000
EVENT_LOG_MAX = 200
