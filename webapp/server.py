"""
Reflex Guard — real-time dashboard server.

Architecture:
    Browser A ─┐
    Browser B ─┼── WebSocket (Socket.IO) ── this server ── DataProvider
    Browser C ─┘                                            ├── SimulationProvider (default)
                                                              └── HardwareProvider (future real device)

Every connected browser receives the same broadcast state, so multiple
judges/teammates can watch one dashboard update live in sync — no manual
refresh, no per-browser fake state.

The camera itself is NOT part of this server: it's captured client-side in
the browser via getUserMedia and never leaves the browser (see static/js/app.js).
Only the simulated/real *device* telemetry (AI, safety core, relay, tool)
flows through this server.

Run:
    python server.py
Then open http://localhost:5001 (multiple tabs/machines all see the same state).
"""

import json
import os
import threading
import time

from flask import Flask, render_template, request
from flask_socketio import SocketIO

from simulation_provider import SimulationProvider
from hardware_provider import HardwareProvider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE = os.path.join(BASE_DIR, "data", "events.json")
EVENTS_MAX = 500

app = Flask(__name__)
app.config["SECRET_KEY"] = "reflex-guard-dev"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

_events_lock = threading.Lock()
_events = []


def _load_events():
    global _events
    if os.path.exists(EVENTS_FILE):
        try:
            with open(EVENTS_FILE) as f:
                _events = json.load(f)[-EVENTS_MAX:]
        except (json.JSONDecodeError, OSError):
            _events = []


def _save_events():
    try:
        with open(EVENTS_FILE, "w") as f:
            json.dump(_events[-EVENTS_MAX:], f)
    except OSError:
        pass


def emit_state(state):
    socketio.emit("state_update", state)


def emit_event(entry):
    with _events_lock:
        _events.append(entry)
        del _events[:-EVENTS_MAX]
        _save_events()
    socketio.emit("event_log", entry)


def emit_log(message):
    socketio.emit("system_log", {"time": time.strftime("%H:%M:%S"), "message": message})


_load_events()

providers = {
    "simulation": SimulationProvider(emit_state, emit_event, emit_log),
    "live": HardwareProvider(emit_state, emit_event, emit_log),
}

# PC LIVE mode (Snapdragon PC + Arduino UNO Q over USB serial) needs
# opencv-python and, for a real serial link, pyserial — both optional so a
# plain `pip install -r requirements.txt` still gets you simulation + live
# without pulling in a webcam/serial dependency. See webapp/README.md.
try:
    from pc_provider import PCLiveProvider
    providers["pc"] = PCLiveProvider(emit_state, emit_event, emit_log)
except ImportError as exc:
    print(f"[server] PC LIVE mode unavailable ({exc}). "
          f"`pip install opencv-python pyserial` to enable it.")

current_mode = "simulation"
_mode_lock = threading.Lock()


def active_provider():
    with _mode_lock:
        return providers[current_mode]


# --------------------------------------------------------------- HTTP routes
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed_pc")
def video_feed_pc():
    """MJPEG stream of the PC LIVE provider's annotated camera frames (real
    webcam + real AIHubHandDetector output), used only when mode == 'pc'."""
    from flask import Response

    def gen():
        provider = providers.get("pc")
        while True:
            frame = provider.get_latest_jpeg() if provider else None
            if frame is None:
                time.sleep(0.1)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    if "pc" not in providers:
        return "PC LIVE mode unavailable on this server (missing opencv-python/pyserial).", 503
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------- Socket.IO
@socketio.on("connect")
def on_connect():
    socketio.emit("state_update", active_provider().snapshot(), to=request.sid)
    with _events_lock:
        socketio.emit("event_log_bulk", _events[-100:], to=request.sid)
    emit_log("Client connected to dashboard")


@socketio.on("command")
def on_command(cmd):
    global current_mode
    if cmd.get("action") == "set_mode":
        new_mode = cmd.get("value")
        if new_mode in providers:
            with _mode_lock:
                current_mode = new_mode
            emit_log(f"DATA SOURCE switched to {new_mode.upper()}")
            socketio.emit("state_update", active_provider().snapshot())
        return
    active_provider().handle_command(cmd)


@socketio.on("clear_events")
def on_clear_events():
    with _events_lock:
        _events.clear()
        _save_events()
    socketio.emit("event_log_cleared")
    emit_log("Event log cleared")


# --------------------------------------------------------------- tick loop
def tick_loop():
    while True:
        active_provider().tick()
        time.sleep(0.2)  # 5 Hz


if __name__ == "__main__":
    threading.Thread(target=tick_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
