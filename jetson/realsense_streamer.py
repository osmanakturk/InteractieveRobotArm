# jetson/realsense_streamer.py
from __future__ import annotations

import time
import threading
from typing import Generator, Optional

import cv2
import numpy as np

from config import CONFIG

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None  # type: ignore


class RealSenseMjpegStreamer:
    """
    RealSense color stream -> MJPEG (no detection).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._pipeline: Optional["rs.pipeline"] = None
        self._running = False
        self._enabled = bool(getattr(CONFIG, "camera_autostart", False))
        self._last_err: str = ""

        if self._enabled:
            try:
                self.open()
            except Exception:
                pass

    def last_error(self) -> str:
        with self._lock:
            return self._last_err

    def _set_err(self, msg: str) -> None:
        with self._lock:
            self._last_err = msg

    def is_started(self) -> bool:
        with self._lock:
            return bool(self._enabled and self._running)

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

                cfg.enable_stream(
                    rs.stream.color,
                    CONFIG.rs_width,
                    CONFIG.rs_height,
                    rs.format.bgr8,
                    CONFIG.rs_fps,
                )

                pipeline.start(cfg)

                self._pipeline = pipeline
                self._running = True
                self._set_err("")
                return True
            except Exception as e:
                self._pipeline = None
                self._running = False
                self._set_err(f"RealSense open failed: {e}")
                return False

    def close(self) -> None:
        with self._lock:
            p = self._pipeline
            self._pipeline = None
            self._running = False

        if p is not None:
            try:
                p.stop()
            except Exception:
                pass

    def _make_info_frame(self, text: str) -> bytes:
        w, h = CONFIG.rs_width, CONFIG.rs_height
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Centered-ish text
        cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)],
        )
        return jpg.tobytes() if ok else b""

    @staticmethod
    def _wrap_jpg(jpg_bytes: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"

    def frames(self) -> Generator[bytes, None, None]:
        """
        IMPORTANT:
        - Must react to stop()/start() while the client keeps the same HTTP connection open.
        - Use poll_for_frames() to avoid blocking forever (so stop works instantly).
        """
        info_stopped = self._make_info_frame("Camera is OFF. Press Cam-On to start.")
        last_info_sent_at = 0.0

        while True:
            with self._lock:
                enabled = bool(self._enabled)
                running = bool(self._running)
                pipeline = self._pipeline

            # If disabled: ensure closed and keep yielding info frames
            if not enabled:
                if running or pipeline is not None:
                    self.close()

                now = time.monotonic()
                # avoid re-encoding constantly
                if now - last_info_sent_at > 2.0:
                    info_stopped = self._make_info_frame("Camera is OFF. Press Cam-On to start.")
                    last_info_sent_at = now

                yield self._wrap_jpg(info_stopped)
                time.sleep(0.2)
                continue

            # Enabled but not running: try open
            if not running or pipeline is None:
                ok = self.open()
                if not ok:
                    err = self.last_error() or "RealSense not available"
                    jpg = self._make_info_frame(err)
                    yield self._wrap_jpg(jpg)
                    time.sleep(0.25)
                    continue

                with self._lock:
                    pipeline = self._pipeline

            if pipeline is None:
                jpg = self._make_info_frame("Camera pipeline missing.")
                yield self._wrap_jpg(jpg)
                time.sleep(0.25)
                continue

            try:
                # Non-blocking fetch -> allows stop() to take effect immediately
                frameset = pipeline.poll_for_frames()
                if not frameset:
                    time.sleep(0.005)
                    continue

                color_frame = frameset.get_color_frame()
                if not color_frame:
                    time.sleep(0.005)
                    continue

                color_image = np.asanyarray(color_frame.get_data())  # BGR

                ok, jpg = cv2.imencode(
                    ".jpg",
                    color_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)],
                )
                if not ok:
                    continue

                yield self._wrap_jpg(jpg.tobytes())
                time.sleep(1.0 / max(1, int(CONFIG.rs_fps)))

            except GeneratorExit:
                return
            except Exception as e:
                # Mark error and fall back to loop (will reopen or show error)
                self._set_err(f"stream error: {e}")
                self.close()
                time.sleep(0.1)
                continue