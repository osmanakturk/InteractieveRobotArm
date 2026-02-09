# jetson/main.py
from __future__ import annotations

import asyncio
import platform
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import CONFIG
from robot_controller import RobotController
from realsense_streamer import RealSenseDetectorStreamer


# -------------------------
# Windows: reduce noisy ConnectionResetError (WinError 10054)
# -------------------------
if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass


# -------------------------
# Helpers (JSON)
# -------------------------
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


def normalize_any_http(input_value: str) -> str:
    """
    Accepts:
      - "192.168.1.20:9000"
      - "http://192.168.1.20:9000"
      - "https://192.168.1.20:9000"
    Returns a valid http(s) URL.
    Rule: if no scheme -> prepend http://
    """
    raw = (input_value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw
    return f"http://{raw}"


# -------------------------
# Pydantic bodies
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
    url: str  # can be ip:port or http(s)://...


# -------------------------
# AI Server Monitor
# -------------------------
class AiMonitor:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._configured_url: Optional[str] = None
        self._connected: bool = False
        self._last_check_ts: Optional[float] = None
        self._last_latency_ms: Optional[int] = None
        self._last_error: str = ""

        self._client: Optional[httpx.AsyncClient] = None
        self._last_logged_connected: Optional[bool] = None

    async def set_client(self, client: httpx.AsyncClient) -> None:
        async with self._lock:
            self._client = client

    async def configure_and_connect(self, url: str) -> Tuple[bool, str]:
        url = normalize_any_http(url)
        if not url:
            return False, "url is empty"

        async with self._lock:
            self._configured_url = url
            self._connected = False
            self._last_error = ""
            self._last_latency_ms = None
            self._last_check_ts = None
            self._last_logged_connected = None

        print(f"[AI] configured url -> {url}")
        ok1, msg = await self.check_once()
        if ok1:
            print(f"[AI] CONNECTED -> {url}")
            return True, "AI server connected."
        print(f"[AI] connect failed -> {msg}")
        return False, msg

    async def disconnect(self) -> None:
        async with self._lock:
            if self._configured_url:
                print(f"[AI] DISCONNECTED -> {self._configured_url}")
            self._configured_url = None
            self._connected = False
            self._last_error = ""
            self._last_latency_ms = None
            self._last_check_ts = None
            self._last_logged_connected = None

    async def check_once(self) -> Tuple[bool, str]:
        async with self._lock:
            url = self._configured_url
            client = self._client

        if not url:
            async with self._lock:
                self._connected = False
            return False, "AI server not configured."

        if client is None:
            return False, "Internal client not ready."

        health_url = url.rstrip("/") + "/health"
        t0 = time.perf_counter()
        try:
            r = await client.get(health_url)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            dt_ms = int((time.perf_counter() - t0) * 1000)

            if r.status_code == 200 and bool(data.get("ok")):
                async with self._lock:
                    self._connected = True
                    self._last_error = ""
                    self._last_latency_ms = dt_ms
                    self._last_check_ts = time.time()
                return True, "OK"

            msg = f"Health not OK (http={r.status_code})"
            async with self._lock:
                self._connected = False
                self._last_error = msg
                self._last_latency_ms = dt_ms
                self._last_check_ts = time.time()
            return False, msg

        except Exception as e:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            msg = f"Health check failed: {e}"
            async with self._lock:
                self._connected = False
                self._last_error = msg
                self._last_latency_ms = dt_ms
                self._last_check_ts = time.time()
            return False, msg

    async def status(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "configured": bool(self._configured_url),
                "url": self._configured_url,
                "connected": self._connected,
                "last_check_ts": self._last_check_ts,
                "latency_ms": self._last_latency_ms,
                "error": self._last_error,
            }

    async def log_on_change(self) -> None:
        st = await self.status()
        connected = bool(st.get("connected"))
        configured = bool(st.get("configured"))
        if not configured:
            self._last_logged_connected = None
            return
        if self._last_logged_connected is None or self._last_logged_connected != connected:
            self._last_logged_connected = connected
            if connected:
                print(f"[AI] health OK ({st.get('latency_ms')}ms) -> {st.get('url')}")
            else:
                print(f"[AI] health FAIL -> {st.get('error')}")


# -------------------------
# App singletons
# -------------------------
robot = RobotController()
camera = RealSenseDetectorStreamer()
ai = AiMonitor()

_ai_task: Optional[asyncio.Task] = None
_http_client: Optional[httpx.AsyncClient] = None


async def _ai_watchdog_loop() -> None:
    while True:
        try:
            st = await ai.status()
            if st.get("configured"):
                await ai.check_once()
                await ai.log_on_change()
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[AI] watchdog error: {e}")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ai_task, _http_client

    _http_client = httpx.AsyncClient(timeout=1.5)
    await ai.set_client(_http_client)

    _ai_task = asyncio.create_task(_ai_watchdog_loop())

    try:
        yield
    finally:
        # cancel watchdog
        if _ai_task:
            _ai_task.cancel()
            try:
                await _ai_task
            except Exception:
                pass
            _ai_task = None

        # hard cleanup
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
            _http_client = None


app = FastAPI(title="Jetson Gateway API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Status
# -------------------------
@app.get("/api/status")
async def api_status():
    return ok(
        {
            "status": robot.status(),
            "safety": robot.safety_status(),
            "camera": {"started": camera.is_started(), "last_error": camera.last_error()},
            "ai_server": await ai.status(),
        }
    )


# -------------------------
# Gateway hard disconnect (server-side reset)
# -------------------------
@app.post("/api/gateway/disconnect")
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

    print("[GW] gateway reset: robot/camera/ai cleared")
    return ok(
        {
            "message": "Gateway reset done.",
            "ai_server": await ai.status(),
            "status": robot.status(),
            "camera": {"started": camera.is_started(), "last_error": camera.last_error()},
        }
    )


# -------------------------
# Robot API
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
    success, msg, throttled = robot.jog_delta(
        body.dx, body.dy, body.dz, body.droll, body.dpitch, body.dyaw
    )
    if not success:
        if throttled:
            return fail(msg, status=429, data={"throttled": True})
        return fail(msg)
    return ok({"message": msg})


@app.get("/api/gripper/status")
def api_gripper_status():
    return ok({"gripper": robot.gripper_status()})


@app.post("/api/gripper")
def api_gripper(body: GripperBody):
    action = (body.action or "").strip().lower()
    success, msg = robot.gripper_jog(action)
    if not success:
        return fail(msg)
    return ok({"message": msg, "status": robot.status(), "gripper": robot.gripper_status()})


@app.post("/api/safety/clear")
def api_safety_clear():
    robot.clear_safety()
    return ok({"message": "Safety status cleared.", "safety": robot.safety_status()})


# -------------------------
# AI Server API (connect/disconnect)
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
# Camera API
# -------------------------
@app.post("/api/camera/realsense/start")
def api_camera_start():
    started = camera.start()
    if not started:
        return fail(camera.last_error() or "Camera start failed.")
    return ok({"message": "RealSense started.", "camera": {"started": camera.is_started()}})


@app.post("/api/camera/realsense/stop")
def api_camera_stop():
    camera.stop()
    return ok({"message": "RealSense stopped.", "camera": {"started": camera.is_started()}})


@app.get("/api/camera/realsense/status")
def api_camera_status():
    return ok({"camera": {"started": camera.is_started(), "last_error": camera.last_error()}})


@app.get("/api/camera/realsense")
def api_camera_realsense():
    return StreamingResponse(
        camera.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# -------------------------
# Websocket status
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
                    "camera": {"started": camera.is_started(), "last_error": camera.last_error()},
                    "ai_server": await ai.status(),
                }
            )
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__ == "__main__":
    # access_log çok spam oluyorsa access_log=False yapabilirsin.
    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        access_log=True,
        timeout_graceful_shutdown=2,  # kapanmayı beklerken takılmasın
    )
