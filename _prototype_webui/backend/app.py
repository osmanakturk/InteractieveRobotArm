# backend/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory, render_template, Response
from flask_socketio import SocketIO


from config import CONFIG
from robot_controller import RobotController
from camera import CameraStreamer


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_TEMPLATES = ROOT / "frontend" / "templates"
FRONTEND_STATIC = ROOT / "frontend" / "static"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_TEMPLATES),
    static_folder=str(FRONTEND_STATIC),
    static_url_path="/static",
)
socketio = SocketIO(app=app)




robot = RobotController()
camera = CameraStreamer()


# -------------------------
# Pages
# -------------------------
@app.get("/")
def index():
    return render_template("index.html")


# -------------------------
# Camera stream
# -------------------------
@app.get("/camera")
def camera_stream():
    return Response(camera.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# -------------------------
# API helpers
# -------------------------
def ok(data: Dict[str, Any] | None = None):
    payload = {"ok": True}
    if data:
        payload.update(data)
    return jsonify(payload)

def fail(message: str, data: Dict[str, Any] | None = None, status: int = 400):
    payload = {"ok": False, "message": message}
    if data:
        payload.update(data)
    return jsonify(payload), status


# -------------------------
# API: connection & state
# -------------------------
@app.post("/api/connect")
def api_connect():
    body = request.get_json(silent=True) or {}
    ip = (body.get("ip") or "").strip()
    success, msg = robot.connect(ip)
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})

@app.post("/api/disconnect")
def api_disconnect():
    success, msg = robot.disconnect()
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})

@app.post("/api/enable")
def api_enable():
    success, msg = robot.enable()
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})

@app.post("/api/disable")
def api_disable():
    success, msg = robot.disable()
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})

@app.post("/api/stop")
def api_stop():
    success, msg = robot.stop()
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})

@app.post("/api/home")
def api_home():
    success, msg = robot.go_home()
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})


# -------------------------
# API: jog config
# -------------------------
@app.post("/api/frame")
def api_frame():
    body = request.get_json(silent=True) or {}
    frame = (body.get("frame") or "").strip().lower()
    success, msg = robot.set_frame(frame)
    if not success:
        return fail(msg)
    return ok({"message": msg})

@app.post("/api/speed")
def api_speed():
    body = request.get_json(silent=True) or {}
    try:
        pct = int(body.get("speed_pct"))
    except Exception:
        return fail("speed_pct must be int 1..100")
    success, msg = robot.set_speed_pct(pct)
    if not success:
        return fail(msg)
    return ok({"message": msg})


# -------------------------
# API: jog (hold)
# -------------------------
@app.post("/api/jog")
def api_jog():
    body = request.get_json(silent=True) or {}
    try:
        dx = float(body.get("dx", 0.0))
        dy = float(body.get("dy", 0.0))
        dz = float(body.get("dz", 0.0))
        droll = float(body.get("droll", 0.0))
        dpitch = float(body.get("dpitch", 0.0))
        dyaw = float(body.get("dyaw", 0.0))
    except Exception:
        return fail("Invalid jog payload.")

    success, msg = robot.jog_delta(dx, dy, dz, droll, dpitch, dyaw)
    if not success:
        return fail(msg)
    return ok({"message": msg})


# -------------------------
# API: gripper (hold)
# -------------------------
@app.post("/api/gripper")
def api_gripper():
    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "").strip().lower()
    success, msg = robot.gripper_jog(action)
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status()})


# -------------------------
# API: telemetry
# -------------------------
@app.get("/api/status")
def api_status():
    return ok({"status": robot.status()})


if __name__ == "__main__":
    #app.run(host=CONFIG.host, port=CONFIG.port, debug=CONFIG.debug)
    socketio.run(host=CONFIG.host, port=CONFIG.port, debug=CONFIG.debug)
