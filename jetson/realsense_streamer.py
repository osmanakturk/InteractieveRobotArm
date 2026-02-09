# jetson/realsense_streamer.py
from __future__ import annotations

import time
import threading
from typing import Generator, Optional

import numpy as np
import cv2

from config import CONFIG

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore


class RealSenseDetectorStreamer:
    """
    RealSense color+depth -> align depth to color -> YOLO seg -> overlay masks + contour + depth label -> MJPEG.
    Uses CONFIG only.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._pipeline: Optional["rs.pipeline"] = None
        self._align: Optional["rs.align"] = None
        self._depth_scale: float = 0.0

        self._model = None
        self._colors = None  # np array (80,3)

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
            if YOLO is None:
                self._set_err("ultralytics not installed.")
                return False

            try:
                if self._model is None:
                    self._model = YOLO(CONFIG.yolo_model_path)

                if self._colors is None:
                    rng = np.random.default_rng(42)
                    self._colors = rng.integers(0, 255, size=(80, 3), dtype=np.uint8)

                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, CONFIG.rs_width, CONFIG.rs_height, rs.format.z16, CONFIG.rs_fps)
                cfg.enable_stream(rs.stream.color, CONFIG.rs_width, CONFIG.rs_height, rs.format.bgr8, CONFIG.rs_fps)
                profile = pipeline.start(cfg)

                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())

                self._align = rs.align(rs.stream.color)
                self._pipeline = pipeline

                self._running = True
                self._set_err("")
                return True
            except Exception as e:
                self._pipeline = None
                self._align = None
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
            self._align = None
            self._running = False

    def _make_error_frame(self, text: str) -> bytes:
        w, h = CONFIG.rs_width, CONFIG.rs_height
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
        return jpg.tobytes() if ok else b""

    def frames(self) -> Generator[bytes, None, None]:
        # If not enabled, keep yielding a static info frame
        with self._lock:
            enabled = self._enabled

        if not enabled:
            jpg = self._make_error_frame("RealSense stream is stopped. Press Start.")
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.2)

        if not self.open():
            err = self.last_error() or "RealSense/YOLO not available"
            jpg = self._make_error_frame(err)
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.2)

        assert self._pipeline is not None
        assert self._align is not None
        assert self._model is not None
        assert self._colors is not None

        kernel = int(CONFIG.depth_kernel)
        if kernel < 1 or kernel % 2 == 0:
            kernel = 5
        offset = kernel // 2

        try:
            while True:
                frameset = self._pipeline.wait_for_frames()
                aligned = self._align.process(frameset)

                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    time.sleep(0.005)
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())  # uint16
                color_image = np.asanyarray(color_frame.get_data())  # bgr

                rows, cols, _ = color_image.shape
                overlay = color_image.copy()

                results = self._model(color_image, stream=True, verbose=False, retina_masks=True)

                for r in results:
                    if r.masks is None:
                        continue

                    for box, mask_poly in zip(r.boxes, r.masks.xy):
                        cls_id = int(box.cls[0]) if box.cls is not None else 0
                        cls_id = max(0, min(79, cls_id))
                        color = tuple(int(c) for c in self._colors[cls_id].tolist())
                        class_name = self._model.names.get(cls_id, str(cls_id))

                        contour = mask_poly.astype(np.int32).reshape(-1, 1, 2)

                        cv2.fillPoly(overlay, [contour], color)
                        cv2.drawContours(color_image, [contour], -1, color, 2)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        min_x = max(0, cx - offset)
                        max_x = min(cols, cx + offset + 1)
                        min_y = max(0, cy - offset)
                        max_y = min(rows, cy + offset + 1)

                        depth_roi = depth_image[min_y:max_y, min_x:max_x]
                        valid = depth_roi[depth_roi > 0]

                        if valid.size > 0:
                            dist_raw = float(np.mean(valid))
                            dist_m = dist_raw * self._depth_scale
                            dist_text = f"{dist_m:.2f}m"
                        else:
                            dist_text = "?.??m"

                        cv2.rectangle(color_image, (min_x, min_y), (max_x, max_y), (255, 255, 255), 1)

                        label = f"{class_name} {dist_text}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(color_image, (cx - 5, cy - th - 8), (cx + tw + 8, cy + 6), (0, 0, 0), -1)
                        cv2.putText(color_image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                alpha = float(CONFIG.overlay_alpha)
                alpha = max(0.0, min(1.0, alpha))
                cv2.addWeighted(overlay, alpha, color_image, 1.0 - alpha, 0.0, color_image)

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
