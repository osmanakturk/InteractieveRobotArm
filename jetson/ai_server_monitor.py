# jetson/ai_server_monitor.py
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Tuple

import httpx


def normalize_any_http(input_value: str) -> str:
    raw = (input_value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw.rstrip("/")
    return f"http://{raw}".rstrip("/")


class AiMonitor:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._configured_url: Optional[str] = None
        self._connected: bool = False
        self._last_check_ts: Optional[float] = None
        self._last_latency_ms: Optional[int] = None
        self._last_error: str = ""
        self._modes: Optional[Dict[str, Any]] = None

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
            self._modes = None
            self._last_logged_connected = None

        ok1, msg = await self.check_once()
        if ok1:
            return True, "AI server connected."
        return False, msg

    async def disconnect(self) -> None:
        async with self._lock:
            self._configured_url = None
            self._connected = False
            self._last_error = ""
            self._last_latency_ms = None
            self._last_check_ts = None
            self._modes = None
            self._last_logged_connected = None

    async def check_once(self) -> Tuple[bool, str]:
        async with self._lock:
            url = self._configured_url
            client = self._client

        if not url:
            async with self._lock:
                self._connected = False
                self._modes = None
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
                modes = None
                # optional: modes status
                try:
                    mr = await client.get(url.rstrip("/") + "/api/ai_server/modes/status")
                    if mr.status_code == 200:
                        modes = mr.json()
                except Exception:
                    modes = None

                async with self._lock:
                    self._connected = True
                    self._last_error = ""
                    self._last_latency_ms = dt_ms
                    self._last_check_ts = time.time()
                    self._modes = modes
                return True, "OK"

            msg = f"Health not OK (http={r.status_code})"
            async with self._lock:
                self._connected = False
                self._last_error = msg
                self._last_latency_ms = dt_ms
                self._last_check_ts = time.time()
                self._modes = None
            return False, msg

        except Exception as e:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            msg = f"Health check failed: {e}"
            async with self._lock:
                self._connected = False
                self._last_error = msg
                self._last_latency_ms = dt_ms
                self._last_check_ts = time.time()
                self._modes = None
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
                "modes": self._modes,
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


async def ai_watchdog_loop(ai: AiMonitor, interval_s: float = 1.0) -> None:
    while True:
        try:
            st = await ai.status()
            if st.get("configured"):
                await ai.check_once()
                await ai.log_on_change()
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[AI] watchdog error: {e}")
            await asyncio.sleep(interval_s)