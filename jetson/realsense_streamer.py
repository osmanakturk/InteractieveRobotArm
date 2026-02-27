from __future__ import annotations

import base64
import time
import threading
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from config import CONFIG

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None  # type: ignore


@dataclass(frozen=True)
class RsIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class RealSenseMjpegStreamer:
    """
    RealSense:
      - Captures COLOR + DEPTH
      - Aligns DEPTH -> COLOR
      - Keeps latest frames in memory (thread-safe)
      - Serves MJPEG of color
      - Provides get_grasp_pack(): aligned depth crop encoded as PNG base64
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._pipeline: Optional["rs.pipeline"] = None
        self._align: Optional["rs.align"] = None

        self._enabled: bool = bool(getattr(CONFIG, "camera_autostart", False))
        self._running: bool = False

        self._last_err: str = ""
        self._depth_scale: float = 0.0
        self._intr: Optional[RsIntrinsics] = None

        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth_u16: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0  # monotonic

        self._grab_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Autostart if enabled
        if self._enabled:
            self.start()

    # -------------------------
    # Status / info
    # -------------------------
    def last_error(self) -> str:
        with self._lock:
            return self._last_err

    def _set_err(self, msg: str) -> None:
        with self._lock:
            self._last_err = msg

    def is_started(self) -> bool:
        with self._lock:
            return bool(self._enabled and self._running and self._pipeline is not None)

    def get_depth_scale(self) -> float:
        with self._lock:
            return float(self._depth_scale)

    def get_intrinsics(self) -> Optional[RsIntrinsics]:
        with self._lock:
            return self._intr

    def last_frame_age_s(self) -> Optional[float]:
        with self._lock:
            if self._last_frame_ts <= 0:
                return None
            return float(time.monotonic() - self._last_frame_ts)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (color_bgr, depth_u16_aligned_to_color).
        Copies are returned to avoid concurrent mutation issues.
        """
        with self._lock:
            c = None if self._last_color_bgr is None else self._last_color_bgr.copy()
            d = None if self._last_depth_u16 is None else self._last_depth_u16.copy()
            return c, d

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self) -> bool:
        with self._lock:
            self._enabled = True
        return self.open()

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
        self.close()

    def open(self) -> bool:
        with self._lock:
            if self._running and self._pipeline is not None:
                return True

            if rs is None:
                self._set_err("pyrealsense2 not installed.")
                return False

            try:
                pipeline = rs.pipeline()
                cfg = rs.config()

                # We need both color and depth for pick&place
                cfg.enable_stream(rs.stream.color, CONFIG.rs_width, CONFIG.rs_height, rs.format.bgr8, CONFIG.rs_fps)
                cfg.enable_stream(rs.stream.depth, CONFIG.rs_width, CONFIG.rs_height, rs.format.z16, CONFIG.rs_fps)

                profile = pipeline.start(cfg)

                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = float(depth_sensor.get_depth_scale())

                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_stream.get_intrinsics()

                self._pipeline = pipeline
                self._align = rs.align(rs.stream.color)  # align depth -> color
                self._depth_scale = depth_scale
                self._intr = RsIntrinsics(
                    fx=float(intr.fx),
                    fy=float(intr.fy),
                    ppx=float(intr.ppx),
                    ppy=float(intr.ppy),
                    width=int(intr.width),
                    height=int(intr.height),
                )

                self._stop_evt.clear()
                self._running = True
                self._set_err("")

                # Start grab thread
                if self._grab_thread is None or not self._grab_thread.is_alive():
                    self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
                    self._grab_thread.start()

                return True

            except Exception as e:
                self._pipeline = None
                self._align = None
                self._running = False
                self._depth_scale = 0.0
                self._intr = None
                self._set_err(f"RealSense open failed: {e}")
                return False

    def close(self) -> None:
        with self._lock:
            p = self._pipeline
            self._pipeline = None
            self._align = None
            self._running = False

            # stop thread loop
            self._stop_evt.set()

            # keep last frames (optional), but reset timestamps so status shows "stale"
            self._last_frame_ts = 0.0

        if p is not None:
            try:
                p.stop()
            except Exception:
                pass

    def _grab_loop(self) -> None:
        """
        Continuously grabs frames while enabled+running.
        Uses wait_for_frames() but we manage stop via pipeline stop + event.
        """
        while not self._stop_evt.is_set():
            with self._lock:
                enabled = bool(self._enabled)
                running = bool(self._running)
                pipeline = self._pipeline
                align = self._align

            if not enabled:
                time.sleep(0.05)
                continue

            if not running or pipeline is None or align is None:
                # Try reopening if enabled
                ok = self.open()
                if not ok:
                    time.sleep(0.25)
                continue

            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
                aligned = align.process(frames)

                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_u16 = np.asanyarray(depth_frame.get_data())  # uint16 (aligned to color)
                color_bgr = np.asanyarray(color_frame.get_data())  # bgr8

                with self._lock:
                    self._last_depth_u16 = depth_u16
                    self._last_color_bgr = color_bgr
                    self._last_frame_ts = time.monotonic()

            except Exception as e:
                self._set_err(f"grab error: {e}")
                # Force reopen path
                try:
                    self.close()
                except Exception:
                    pass
                time.sleep(0.1)

    # -------------------------
    # MJPEG stream
    # -------------------------
    def _make_info_frame(self, text: str) -> bytes:
        w, h = CONFIG.rs_width, CONFIG.rs_height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
        return jpg.tobytes() if ok else b""

    @staticmethod
    def _wrap_jpg(jpg_bytes: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"

    def frames(self) -> Generator[bytes, None, None]:
        """
        MJPEG stream of COLOR.
        - If disabled: yields info frame.
        - If enabled but not ready: yields error/info frame.
        """
        info_off = self._make_info_frame("Camera is OFF. Call /api/cameras/realsense/start")
        last_info_sent_at = 0.0

        while True:
            with self._lock:
                enabled = bool(self._enabled)
                running = bool(self._running)
                color = None if self._last_color_bgr is None else self._last_color_bgr.copy()
                err = self._last_err

            if not enabled:
                now = time.monotonic()
                if now - last_info_sent_at > 2.0:
                    info_off = self._make_info_frame("Camera is OFF. Call /api/cameras/realsense/start")
                    last_info_sent_at = now
                yield self._wrap_jpg(info_off)
                time.sleep(0.2)
                continue

            if not running:
                jpg = self._make_info_frame(err or "Camera starting...")
                yield self._wrap_jpg(jpg)
                time.sleep(0.25)
                continue

            if color is None:
                jpg = self._make_info_frame("Waiting for frames...")
                yield self._wrap_jpg(jpg)
                time.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
            if ok:
                yield self._wrap_jpg(jpg.tobytes())

            time.sleep(1.0 / max(1, int(CONFIG.rs_fps)))

    # -------------------------
    # Pick&Place: grasp pack
    # -------------------------
    def get_grasp_pack(self, cx: int, cy: int, crop_size: int, *, auto_start: bool = True) -> Dict[str, Any]:
        """
        Returns payload compatible with old Mac code:
          {
            "valid": True,
            "depth_crop_b64": <png base64 (uint16 depth crop)>,
            "crop_origin": {"x": x1, "y": y1},
            "crop_size": {"w": w, "h": h},
            "depth_scale": <float>,
            "intrinsics": {"fx","fy","ppx","ppy"}
          }

        If not ready => {"valid": False, "error": "..."} (caller decides HTTP status)
        """

        if auto_start and not self.is_started():
            ok = self.start()
            if not ok:
                return {"valid": False, "error": self.last_error() or "Camera is OFF and failed to start."}

            # tiny warmup for first frames
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                _, d = self.get_latest()
                if d is not None:
                    break
                time.sleep(0.02)

        intr = self.get_intrinsics()
        depth_scale = self.get_depth_scale()
        _, depth_u16 = self.get_latest()

        if intr is None or depth_u16 is None or depth_scale <= 0:
            return {"valid": False, "error": self.last_error() or "Depth/intrinsics not ready."}

        h, w = depth_u16.shape[:2]

        cx = _clamp_int(int(cx), 0, w - 1)
        cy = _clamp_int(int(cy), 0, h - 1)

        crop_size = int(crop_size)
        crop_size = _clamp_int(crop_size, 64, min(w, h))
        half = crop_size // 2

        x1 = _clamp_int(cx - half, 0, w - 1)
        y1 = _clamp_int(cy - half, 0, h - 1)
        x2 = _clamp_int(cx + half, 1, w)
        y2 = _clamp_int(cy + half, 1, h)

        if x2 <= x1 or y2 <= y1:
            return {"valid": False, "error": "Invalid crop window."}

        depth_crop = depth_u16[y1:y2, x1:x2]  # uint16 aligned-to-color

        ok_png, png = cv2.imencode(".png", depth_crop)
        if not ok_png:
            return {"valid": False, "error": "PNG encode failed."}

        depth_b64 = base64.b64encode(png.tobytes()).decode("utf-8")

        return {
            "valid": True,
            "depth_crop_b64": depth_b64,
            "crop_origin": {"x": int(x1), "y": int(y1)},
            "crop_size": {"w": int(depth_crop.shape[1]), "h": int(depth_crop.shape[0])},
            "depth_scale": float(depth_scale),
            "intrinsics": {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
            },
        }