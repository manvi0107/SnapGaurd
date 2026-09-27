"""
Reflex Guard dashboard.

Runs the vision loop in a background thread, feeds every frame's danger
flag into the RealtimeCore (the simulated STM32 safety layer), and serves
a live MJPEG feed + JSON status to the browser.

Run:
    python -m src.dashboard.app
Then open http://<pi-ip>:5000
"""

import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src import config  # noqa: E402
from src.vision.detector import CameraSource, SimulatedSource, build_detector  # noqa: E402
from src.safety.realtime_core import RealtimeCore  # noqa: E402
from src.digital_twin import DigitalTwin  # noqa: E402

app = Flask(__name__)

mode = os.environ.get("REFLEX_MODE", config.MODE)
config.DETECTOR_BACKEND = os.environ.get("REFLEX_DETECTOR", config.DETECTOR_BACKEND)
detector = build_detector(config)
twin = DigitalTwin()
core = RealtimeCore(
    watchdog_timeout_s=config.WATCHDOG_TIMEOUT_S,
    poll_interval_s=config.POLL_INTERVAL_S,
    use_gpio=config.USE_GPIO,
    relay_pin=config.RELAY_PIN,
    event_log_max=config.EVENT_LOG_MAX,
)
core.start()

_latest_jpeg = None
_frame_lock = threading.Lock()


def vision_loop():
    global _latest_jpeg
    if mode == "camera":
        source = CameraSource(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    else:
        source = SimulatedSource(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    try:
        while True:
            frame = source.read()
            annotated, danger, distance_px = detector.process(frame)
            prev_state = core.status()["state"]
            core.signal(danger)
            if distance_px is not None:
                twin.update_hand_position(distance_px)  # px used as a proxy for mm here
            new_status = core.status()
            if prev_state != "TRIPPED" and new_status["state"] == "TRIPPED":
                twin.observe_actual_response(new_status["last_latency_ms"])

            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                with _frame_lock:
                    _latest_jpeg = buf.tobytes()
            time.sleep(0.03)  # ~30 fps cap
    finally:
        source.release()


def mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _latest_jpeg
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html", mode=mode)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(core.status())


@app.route("/api/ai_status")
def api_ai_status():
    return jsonify(detector.status() if hasattr(detector, "status") else {"real_model": False})


@app.route("/api/twin")
def api_twin():
    return jsonify(twin.snapshot())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    core.reset()
    return jsonify({"ok": True})


def main():
    threading.Thread(target=vision_loop, daemon=True).start()
    app.run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
