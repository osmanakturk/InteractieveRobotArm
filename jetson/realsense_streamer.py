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
    You will run detection on AI server later.
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
            return self._enabled and self._running

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
            if self._running:
                return True

            if rs is None:
                self._set_err("pyrealsense2 not installed.")
                return False

            try:
                pipeline = rs.pipeline()
                cfg = rs.config()

                # For now: color only (simple + lower load).
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
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._running = False

    def _make_info_frame(self, text: str) -> bytes:
        w, h = CONFIG.rs_width, CONFIG.rs_height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
        return jpg.tobytes() if ok else b""

    def frames(self) -> Generator[bytes, None, None]:
        # If not enabled, yield static frame
        with self._lock:
            enabled = self._enabled

        if not enabled:
            jpg = self._make_info_frame("RealSense stream is stopped. Press Start.")
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.2)

        if not self.open():
            err = self.last_error() or "RealSense not available"
            jpg = self._make_info_frame(err)
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.2)

        assert self._pipeline is not None

        try:
            while True:
                frameset = self._pipeline.wait_for_frames()
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    time.sleep(0.005)
                    continue

                color_image = np.asanyarray(color_frame.get_data())  # BGR
                ok, jpg = cv2.imencode(".jpg", color_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
                if not ok:
                    continue

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(1.0 / max(1, CONFIG.rs_fps))

        except GeneratorExit:
            pass
        except Exception as e:
            self._set_err(f"stream error: {e}")
        finally:
            pass
