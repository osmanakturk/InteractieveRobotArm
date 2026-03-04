# jetson/main.py
from __future__ import annotations

import asyncio
import platform
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import CONFIG
from robot_controller import RobotController
from realsense_streamer import RealSenseMjpegStreamer
from ai_server_monitor import AiMonitor, ai_watchdog_loop


if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass


def ok(data: Optional[Dict[str, Any]] = None):
    payload: Dict[str, Any] = {"ok": True}
    if data:
        payload.update(data)
    return JSONResponse(payload)


def fail(message: str, status: int = 400, data: Optional[Dict[str, Any]] = None):
    payload: Dict[str, Any] = {"ok": False, "message": message}
    if data:
        payload.update(data)
    return JSONResponse(payload, status_code=status)


class RobotConnectBody(BaseModel):
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


class RobotMovePoseBody(BaseModel):
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    speed: int = Field(120, ge=1, le=2000)
    wait: bool = False


class GripperBody(BaseModel):
    action: str  # open|close


class AiServerBody(BaseModel):
    url: str  # can be ip:port or http(s)://...


class GraspPackBody(BaseModel):
    x: int = Field(..., description="pixel x in color image")
    y: int = Field(..., description="pixel y in color image")
    crop_size: int = Field(300, ge=50, le=1000, description="square crop size")


robot = RobotController()
camera = RealSenseMjpegStreamer()
ai = AiMonitor()

_ai_task: Optional[asyncio.Task] = None
_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ai_task, _http_client

    _http_client = httpx.AsyncClient(timeout=5.0)
    await ai.set_client(_http_client)

    _ai_task = asyncio.create_task(ai_watchdog_loop(ai, interval_s=1.0))

    # optional camera autostart
    try:
        if bool(getattr(CONFIG, "camera_autostart", False)):
            camera.start()
    except Exception:
        pass

    try:
        yield
    finally:
        if _ai_task:
            _ai_task.cancel()
            try:
                await _ai_task
            except Exception:
                pass

        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            camera.stop()
        except Exception:
            pass
        try:
            await ai.disconnect()
        except Exception:
            pass

        if _http_client:
            try:
                await _http_client.aclose()
            except Exception:
                pass


app = FastAPI(title="Jetson Gateway API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# BASIC HEALTH (mobile/monitor expects this)
# -------------------------
@app.get("/health")
def health():
    return ok({"service": "jetson_gateway", "message": "alive"})


@app.get("/api/status")
async def api_status():
    intr = camera.get_intrinsics()
    return ok(
        {
            "robot": robot.status(),
            "safety": robot.safety_status(),
            "cameras": {
                "realsense": {
                    "started": camera.is_started(),
                    "last_error": camera.last_error(),
                    "last_frame_age_s": camera.last_frame_age_s(),
                    "depth_scale": camera.get_depth_scale(),
                    "intrinsics": (intr.__dict__ if intr else None),
                }
            },
            "ai_server": await ai.status(),
        }
    )


@app.post("/api/disconnect")
async def api_gateway_disconnect():
    try:
        robot.disconnect()
    except Exception:
        pass
    try:
        camera.stop()
    except Exception:
        pass
    try:
        await ai.disconnect()
    except Exception:
        pass

    return ok(
        {
            "message": "Gateway reset done.",
            "ai_server": await ai.status(),
            "robot": robot.status(),
            "cameras": {"realsense": {"started": camera.is_started(), "last_error": camera.last_error()}},
        }
    )


# -------------------------
# ROBOT
# -------------------------
@app.post("/api/robot/connect")
def api_robot_connect(body: RobotConnectBody):
    success, msg = robot.connect(body.ip)
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/disconnect")
def api_robot_disconnect():
    success, msg = robot.disconnect()
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/enable")
def api_robot_enable():
    success, msg = robot.enable()
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/disable")
def api_robot_disable():
    success, msg = robot.disable()
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/stop")
def api_robot_stop():
    success, msg = robot.stop()
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/home")
def api_robot_home():
    success, msg = robot.go_home()
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.post("/api/robot/frame")
def api_robot_frame(body: FrameBody):
    success, msg = robot.set_frame(body.frame)
    if not success:
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/robot/speed")
def api_robot_speed(body: SpeedBody):
    success, msg = robot.set_speed_pct(body.speed_pct)
    if not success:
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/robot/jog")
def api_robot_jog(body: JogBody):
    success, msg, throttled = robot.jog_delta(body.dx, body.dy, body.dz, body.droll, body.dpitch, body.dyaw)
    if not success:
        if throttled:
            return fail(msg, status=429, data={"throttled": True})
        return fail(msg)
    return ok({"message": msg})


@app.post("/api/robot/move_pose")
def api_robot_move_pose(body: RobotMovePoseBody):
    success, msg = robot.move_pose(
        x=body.x,
        y=body.y,
        z=body.z,
        roll=body.roll,
        pitch=body.pitch,
        yaw=body.yaw,
        speed=body.speed,
        wait=body.wait,
    )
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status()})


@app.get("/api/robot/gripper/status")
def api_robot_gripper_status():
    return ok({"gripper": robot.gripper_status()})


@app.post("/api/robot/gripper")
def api_robot_gripper(body: GripperBody):
    action = (body.action or "").strip().lower()
    success, msg = robot.gripper_jog(action)
    if not success:
        return fail(msg)
    return ok({"message": msg, "robot": robot.status(), "gripper": robot.gripper_status()})


@app.post("/api/robot/safety/clear")
def api_robot_safety_clear():
    robot.clear_safety()
    return ok({"message": "Safety status cleared.", "safety": robot.safety_status()})


@app.post("/api/robot/vision_pose")
def api_robot_vision_pose():
    vp = CONFIG.robot.vision_pose
    success, msg = robot.move_pose(
        x=vp.x,
        y=vp.y,
        z=vp.z,
        roll=vp.roll,
        pitch=vp.pitch,
        yaw=vp.yaw,
        speed=int(vp.speed),
        wait=bool(vp.wait),
    )
    if not success:
        return fail(msg)
    return ok({"message": "Vision pose reached.", "robot": robot.status()})


# -------------------------
# AI SERVER connect/disconnect
# -------------------------
@app.post("/api/ai_server/connect")
async def api_ai_connect(body: AiServerBody):
    success, msg = await ai.configure_and_connect(body.url)
    if not success:
        return fail(msg)
    return ok({"message": msg, "ai_server": await ai.status()})


@app.post("/api/ai_server/disconnect")
async def api_ai_disconnect():
    await ai.disconnect()
    return ok({"message": "AI server disconnected.", "ai_server": await ai.status()})


# -------------------------
# AI SERVER proxy helpers (mobile expects these on gateway too)
# -------------------------
async def _ai_base_url() -> str:
    # Single truth: AiMonitor.status()["url"] (configured base URL)
    st = await ai.status()
    url = (st.get("url") or "").strip() if isinstance(st, dict) else ""
    return url.rstrip("/") if url else ""


async def _proxy_ai(method: str, path: str, payload: Optional[dict] = None):
    if _http_client is None:
        return fail("HTTP client not ready", status=503)

    base = await _ai_base_url()
    if not base:
        return fail("AI server not connected/configured", status=400)

    url = f"{base}{path}"
    try:
        if method == "GET":
            r = await _http_client.get(url)
        else:
            r = await _http_client.post(url, json=(payload or {}))

        try:
            data = r.json()
        except Exception:
            data = {"ok": False, "message": "invalid JSON from AI server", "raw": r.text}

        return JSONResponse(status_code=r.status_code, content=data)
    except Exception as e:
        return fail(f"AI proxy failed: {e}", status=503)


@app.get("/api/ai_server/modes/status")
async def gw_ai_modes_status():
    return await _proxy_ai("GET", "/api/ai_server/modes/status")


@app.post("/api/ai_server/pick_place/select")
async def gw_pick_place_select(payload: dict = Body(default={})):
    return await _proxy_ai("POST", "/api/ai_server/pick_place/select", payload)


@app.post("/api/ai_server/pick_place/execute")
async def gw_pick_place_execute(payload: dict = Body(default={})):
    return await _proxy_ai("POST", "/api/ai_server/pick_place/execute", payload)


@app.post("/api/ai_server/pick_place/cancel")
async def gw_pick_place_cancel(payload: dict = Body(default={})):
    return await _proxy_ai("POST", "/api/ai_server/pick_place/cancel", payload)


@app.post("/api/ai_server/pick_place/reset")
async def gw_pick_place_reset(payload: dict = Body(default={})):
    return await _proxy_ai("POST", "/api/ai_server/pick_place/reset", payload)


@app.get("/api/ai_server/pick_place/status")
async def gw_pick_place_status():
    return await _proxy_ai("GET", "/api/ai_server/pick_place/status")


# -------------------------
# CAMERAS (RealSense)
# -------------------------
@app.post("/api/cameras/realsense/start")
def api_camera_realsense_start():
    started = camera.start()
    if not started:
        return fail(camera.last_error() or "Camera start failed.", status=503)
    return ok({"message": "RealSense started.", "camera": {"started": camera.is_started()}})


@app.post("/api/cameras/realsense/stop")
def api_camera_realsense_stop():
    camera.stop()
    return ok({"message": "RealSense stopped.", "camera": {"started": camera.is_started()}})


@app.get("/api/cameras/realsense/status")
def api_camera_realsense_status():
    return ok(
        {
            "camera": {
                "started": camera.is_started(),
                "last_error": camera.last_error(),
                "last_frame_age_s": camera.last_frame_age_s(),
            }
        }
    )


@app.get("/api/cameras/realsense/stream/mjpeg")
def api_camera_realsense_stream_mjpeg():
    return StreamingResponse(camera.frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/cameras/realsense/get_grasp_pack")
def api_pick_place_get_grasp_pack(body: GraspPackBody):
    pack = camera.get_grasp_pack(body.x, body.y, body.crop_size, auto_start=True)
    if not pack.get("valid"):
        return JSONResponse(status_code=503, content=pack)
    return pack


# -------------------------
# WEBSOCKET
# -------------------------
@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(
                {
                    "ok": True,
                    "status": robot.status(),
                    "safety": robot.safety_status(),
                    "camera": {
                        "started": camera.is_started(),
                        "last_error": camera.last_error(),
                        "last_frame_age_s": camera.last_frame_age_s(),
                    },
                    "ai_server": await ai.status(),
                }
            )
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        access_log=True,
        timeout_graceful_shutdown=2,
    )