# ai_server/main.py
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import CONFIG


app = FastAPI(title="AI Server", version="1.0.0")


# -------------------------
# Mode process manager
# -------------------------
class ModeProc:
    def __init__(self, mode: str, popen: subprocess.Popen, gateway_url: str, started_at: float) -> None:
        self.mode = mode
        self.popen = popen
        self.gateway_url = gateway_url
        self.started_at = started_at

    def is_running(self) -> bool:
        return self.popen.poll() is None

    def pid(self) -> Optional[int]:
        return self.popen.pid


_active: Optional[ModeProc] = None


def _normalize_any_http(v: str) -> str:
    raw = (v or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw.rstrip("/")
    return f"http://{raw}".rstrip("/")


def _mode_status() -> Dict[str, Any]:
    global _active
    if _active is None:
        return {"active": False, "mode": None}

    running = _active.is_running()
    return {
        "active": bool(running),
        "mode": _active.mode,
        "pid": _active.pid(),
        "gateway_url": _active.gateway_url,
        "started_at": _active.started_at,
        "uptime_s": (time.time() - _active.started_at) if running else None,
        "exit_code": None if running else _active.popen.returncode,
    }


def _stop_active() -> None:
    global _active
    if _active is None:
        return

    p = _active.popen
    try:
        if p.poll() is None:
            # graceful
            if os.name == "nt":
                p.terminate()
            else:
                p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=2.0)
            except Exception:
                p.kill()
    finally:
        _active = None


def _start_mode(mode: str, gateway_url: str) -> Dict[str, Any]:
    global _active

    mode = (mode or "").strip()
    if not mode:
        return {"ok": False, "message": "mode is empty"}

    entry = (CONFIG.mode_entrypoints or {}).get(mode)
    if not entry:
        return {"ok": False, "message": f"unknown mode '{mode}'"}

    gw = _normalize_any_http(gateway_url)
    if not gw:
        return {"ok": False, "message": "gateway_url is empty"}

    if CONFIG.single_active_mode:
        _stop_active()
    else:
        # if multi mode allowed, you'd manage a dict; for now keep simple
        if _active is not None and _active.is_running():
            return {"ok": False, "message": "another mode already running"}

    env = os.environ.copy()
    env["GATEWAY_URL"] = gw

    # run: python -u pick_and_place.py
    cmd = [sys.executable, "-u", entry]
    p = subprocess.Popen(cmd, env=env)

    _active = ModeProc(mode=mode, popen=p, gateway_url=gw, started_at=time.time())
    return {"ok": True, "message": "mode started", "status": _mode_status()}


# -------------------------
# API models
# -------------------------
class ModeStartBody(BaseModel):
    mode: str = Field(..., description="e.g. pick_and_place")
    gateway_url: str = Field(..., description="e.g. http://192.168.1.20:8000")


class ModeStopBody(BaseModel):
    mode: Optional[str] = Field(None, description="optional; if given must match active mode")


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return JSONResponse(
        {
            "ok": True,
            "service": "ai_server",
            "message": CONFIG.health_ok_message,
        }
    )


@app.get("/api/ai_server/modes/status")
def modes_status():
    return JSONResponse(
        {
            "ok": True,
            "server": {"host": CONFIG.host, "port": CONFIG.port, "debug": CONFIG.debug},
            "single_active_mode": CONFIG.single_active_mode,
            "available_modes": sorted(list((CONFIG.mode_entrypoints or {}).keys())),
            "active_mode": _mode_status(),
        }
    )


@app.post("/api/ai_server/modes/start")
def modes_start(body: ModeStartBody):
    res = _start_mode(body.mode, body.gateway_url)
    code = 200 if res.get("ok") else 400
    return JSONResponse(status_code=code, content=res)


@app.post("/api/ai_server/modes/stop")
def modes_stop(body: ModeStopBody):
    global _active
    st = _mode_status()
    if not st.get("active"):
        return JSONResponse({"ok": True, "message": "no active mode"})

    if body.mode and st.get("mode") != body.mode:
        return JSONResponse(status_code=400, content={"ok": False, "message": "mode mismatch"})

    _stop_active()
    return JSONResponse({"ok": True, "message": "stopped"})


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)