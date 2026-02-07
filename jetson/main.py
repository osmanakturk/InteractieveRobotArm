from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CONFIG
from robot_controller import RobotController
from realsense_detector import RealSenseDetectorStreamer
import uvicorn

app = FastAPI(title="Jetson Gateway API", version="1.0.0")

# Basit CORS (index.html başka PC'de çalışacak dedin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ileride kısıtlayabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

robot = RobotController()
camera = RealSenseDetectorStreamer()

AI_SERVER_URL: Optional[str] = None  # gateway üzerinde tutulur


def ok(data: Optional[Dict[str, Any]] = None):
    payload = {"ok": True}
    if data:
        payload.update(data)
    return JSONResponse(payload)


def fail(message: str, status: int = 400, data: Optional[Dict[str, Any]] = None):
    payload = {"ok": False, "message": message}
    if data:
        payload.update(data)
    return JSONResponse(payload, status_code=status)


# -------------------------
# Models
# -------------------------
class ConnectBody(BaseModel):
    ip: str


class FrameBody(BaseModel):
    frame: str  # base|tool


class SpeedBody(BaseModel):
    speed_pct: int  # 1..100


class JogBody(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0


class GripperBody(BaseModel):
    action: str  # open|close


class AiServerBody(BaseModel):
    url: str


# -------------------------
# Health / Status
# -------------------------
@app.get("/api/status")
def api_status():
    return ok({"status": robot.status(), "ai_server": AI_SERVER_URL})


@app.get("/api/gripper/status")
def api_gripper_status():
    return ok({"gripper": robot.gripper_status()})


# -------------------------
# Robot connection & control
# -------------------------
@app.post("/api/connect")
def api_connect(body: ConnectBody):
    success, msg = robot.connect(body.ip)
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


@app.post("/api/frame")
def api_frame(body: FrameBody):
    success, msg = robot.set_frame(body.frame)
    if not success:
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/speed")
def api_speed(body: SpeedBody):
    success, msg = robot.set_speed_pct(body.speed_pct)
    if not success:
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/jog")
def api_jog(body: JogBody):
    success, msg = robot.jog_delta(body.dx, body.dy, body.dz, body.droll, body.dpitch, body.dyaw)
    if not success:
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/gripper")
def api_gripper(body: GripperBody):
    action = (body.action or "").strip().lower()
    success, msg = robot.gripper_jog(action)
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status(), "gripper": robot.gripper_status()})


# -------------------------
# AI server config (gateway stores it; gateway will call it later)
# -------------------------
@app.post("/api/ai_server")
def api_ai_server(body: AiServerBody):
    global AI_SERVER_URL
    url = (body.url or "").strip()
    if not url:
        return fail("url is empty")
    AI_SERVER_URL = url
    return ok({"message": "AI Server set.", "ai_server": AI_SERVER_URL})


# -------------------------
# RealSense camera stream (MJPEG)
# -------------------------
@app.get("/api/camera/realsense")
def api_camera_realsense():
    return StreamingResponse(
        camera.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# -------------------------
# WebSocket: status stream (optional but useful)
# -------------------------
@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"ok": True, "status": robot.status(), "ai_server": AI_SERVER_URL})
            # çok sık olmasın
            await ws.receive_text()  # client "ping" gönderirse akış ilerler
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__=="__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)


    