# jetson/realsense_streamer.py
from __future__ import annotations

import base64
import time
import threading
from dataclasses import dataclass
from typing import Generator, Optional, Dict, Any, Tuple

import cv2
import numpy as np

from config import CONFIG

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None  # type: ignore


@dataclass
class RsIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: Tuple[float, float, float, float, float]


class RealSenseMjpegStreamer:
    """
    RealSense (color + depth aligned to color) -> MJPEG stream + grasp pack endpoint support.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._pipeline: Optional["rs.pipeline"] = None
        self._align: Optional["rs.align"] = None
        self._profile: Optional["rs.pipeline_profile"] = None

        self._enabled: bool = bool(getattr(CONFIG, "camera_autostart", False))
        self._running: bool = False
        self._last_err: str = ""

        # caches
        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth_u16: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

        self._depth_scale: Optional[float] = None
        self._intrinsics: Optional[RsIntrinsics] = None

        if self._enabled:
            self.open()

    # -------------------------
    # Basic getters
    # -------------------------
    def last_error(self) -> str:
        with self._lock:
            return self._last_err

    def _set_err(self, msg: str) -> None:
        with self._lock:
            self._last_err = msg

    def is_started(self) -> bool:
        with self._lock:
            return bool(self._enabled and self._running)

    def last_frame_age_s(self) -> Optional[float]:
        with self._lock:
            ts = self._last_frame_ts
            running = self._running
        if not running or ts <= 0:
            return None
        return max(0.0, time.time() - ts)

    def get_depth_scale(self) -> Optional[float]:
        with self._lock:
            return self._depth_scale

    def get_intrinsics(self) -> Optional[RsIntrinsics]:
        with self._lock:
            return self._intrinsics

    # -------------------------
    # Start/Stop/Open/Close
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

                cfg.enable_stream(rs.stream.color, CONFIG.rs_width, CONFIG.rs_height, rs.format.bgr8, CONFIG.rs_fps)
                cfg.enable_stream(rs.stream.depth, CONFIG.rs_width, CONFIG.rs_height, rs.format.z16, CONFIG.rs_fps)

                profile = pipeline.start(cfg)
                align = rs.align(rs.stream.color)

                # depth scale
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = float(depth_sensor.get_depth_scale())

                # intrinsics from color stream
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_stream.get_intrinsics()
                intrinsics = RsIntrinsics(
                    width=int(intr.width),
                    height=int(intr.height),
                    fx=float(intr.fx),
                    fy=float(intr.fy),
                    ppx=float(intr.ppx),
                    ppy=float(intr.ppy),
                    model=str(intr.model),
                    coeffs=tuple(float(c) for c in intr.coeffs),
                )

                self._pipeline = pipeline
                self._align = align
                self._profile = profile
                self._depth_scale = depth_scale
                self._intrinsics = intrinsics

                self._running = True
                self._set_err("")
                return True

            except Exception as e:
                self._pipeline = None
                self._align = None
                self._profile = None
                self._depth_scale = None
                self._intrinsics = None
                self._running = False
                self._set_err(f"RealSense open failed: {e}")
                return False

    def close(self) -> None:
        with self._lock:
            p = self._pipeline
            self._pipeline = None
            self._align = None
            self._profile = None
            self._running = False

            self._last_color_bgr = None
            self._last_depth_u16 = None
            self._last_frame_ts = 0.0

        if p is not None:
            try:
                p.stop()
            except Exception:
                pass

    # -------------------------
    # Internal capture (non-blocking-ish)
    # -------------------------
    def _poll_update_cache(self) -> bool:
        """
        Pull latest aligned frameset and update caches.
        Returns True if cache updated, False otherwise.
        """
        with self._lock:
            pipeline = self._pipeline
            align = self._align
            enabled = self._enabled
            running = self._running

        if not enabled:
            return False
        if not running or pipeline is None or align is None:
            return False

        try:
            frameset = pipeline.poll_for_frames()
            if not frameset:
                return False

            aligned = align.process(frameset)

            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                return False

            color = np.asanyarray(cf.get_data())  # BGR
            depth = np.asanyarray(df.get_data())  # uint16 (z16)

            with self._lock:
                self._last_color_bgr = color
                self._last_depth_u16 = depth
                self._last_frame_ts = time.time()

            return True

        except Exception as e:
            self._set_err(f"stream error: {e}")
            self.close()
            return False

    # -------------------------
    # MJPEG helpers
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
        info_stopped = self._make_info_frame("Camera is OFF. Press Cam-On to start.")
        last_info_sent_at = 0.0

        while True:
            with self._lock:
                enabled = bool(self._enabled)
                running = bool(self._running)

            if not enabled:
                if running:
                    self.close()

                now = time.monotonic()
                if now - last_info_sent_at > 2.0:
                    info_stopped = self._make_info_frame("Camera is OFF. Press Cam-On to start.")
                    last_info_sent_at = now

                yield self._wrap_jpg(info_stopped)
                time.sleep(0.2)
                continue

            if not running:
                ok = self.open()
                if not ok:
                    err = self.last_error() or "RealSense not available"
                    yield self._wrap_jpg(self._make_info_frame(err))
                    time.sleep(0.25)
                    continue

            updated = self._poll_update_cache()
            if not updated:
                time.sleep(0.005)
                continue

            with self._lock:
                img = self._last_color_bgr

            if img is None:
                time.sleep(0.01)
                continue

            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
            if not ok:
                continue

            yield self._wrap_jpg(jpg.tobytes())
            time.sleep(1.0 / max(1, int(CONFIG.rs_fps)))

    # -------------------------
    # Grasp pack (depth crop) for GGCNN pipeline
    # -------------------------
    def get_grasp_pack(self, x: int, y: int, crop_size: int = 300, auto_start: bool = True) -> Dict[str, Any]:
        """
        Returns:
          {
            "valid": bool,
            "message": str,
            "depth_scale": float,
            "crop_origin": {"x": int, "y": int},
            "depth_crop_b64": str (PNG of uint16 depth crop),
            "intrinsics": {...} or None,
            "ts": float (unix),
          }
        """
        if auto_start:
            with self._lock:
                if not self._enabled:
                    self._enabled = True
            if not self.is_started():
                self.open()

        # try to refresh cache quickly
        t0 = time.time()
        while time.time() - t0 < 0.25:
            if self._poll_update_cache():
                break
            time.sleep(0.005)

        with self._lock:
            if not self._running or self._last_depth_u16 is None:
                return {"valid": False, "message": self._last_err or "Camera/depth not ready."}

            depth_u16 = self._last_depth_u16
            depth_scale = self._depth_scale
            intr = self._intrinsics

        if depth_scale is None:
            return {"valid": False, "message": "Depth scale not available."}

        h, w = depth_u16.shape[:2]
        cs = int(max(10, crop_size))
        half = cs // 2

        cx = int(np.clip(int(x), 0, w - 1))
        cy = int(np.clip(int(y), 0, h - 1))

        x1 = int(np.clip(cx - half, 0, w - 1))
        y1 = int(np.clip(cy - half, 0, h - 1))
        x2 = int(np.clip(x1 + cs, 0, w))
        y2 = int(np.clip(y1 + cs, 0, h))

        # ensure exact size if possible
        crop = depth_u16[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return {"valid": False, "message": "Invalid crop region."}

        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return {"valid": False, "message": "Failed to encode depth crop."}

        return {
            "valid": True,
            "message": "OK",
            "depth_scale": float(depth_scale),
            "crop_origin": {"x": int(x1), "y": int(y1)},
            "depth_crop_b64": base64.b64encode(png.tobytes()).decode("ascii"),
            "intrinsics": (intr.__dict__ if intr else None),
            "ts": time.time(),
        }