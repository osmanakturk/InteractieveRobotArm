from __future__ import annotations

import time
import threading
from typing import Generator, Optional

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

from config import CONFIG


class RealSenseDetectorStreamer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running = False

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._depth_scale: float = 0.001  # fallback

        self._model: Optional[YOLO] = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            # YOLO
            self._model = YOLO(CONFIG.yolo_model_path)

            # RealSense pipeline
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

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._align = None
            self._running = False

    def _get_frames(self):
        assert self._pipeline is not None
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        return depth, color

    def _depth_at_center(self, depth_img: np.ndarray, cx: int, cy: int) -> Optional[float]:
        k = max(1, int(CONFIG.depth_kernel))
        if k % 2 == 0:
            k += 1
        off = k // 2

        h, w = depth_img.shape[:2]
        x1 = max(0, cx - off)
        x2 = min(w, cx + off + 1)
        y1 = max(0, cy - off)
        y2 = min(h, cy + off + 1)

        roi = depth_img[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None

        dist_raw = float(np.mean(valid))
        return dist_raw * self._depth_scale

    def frames(self) -> Generator[bytes, None, None]:
        """
        MJPEG generator: /api/camera/realsense
        """
        # Lazy start
        try:
            self.start()
        except Exception:
            # fallback frame
            img = np.zeros((CONFIG.rs_height, CONFIG.rs_width, 3), dtype=np.uint8)
            cv2.putText(img, "RealSense/YOLO not available", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, jpg = cv2.imencode(".jpg", img)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            return

        assert self._model is not None

        while True:
            with self._lock:
                if not self._running or self._pipeline is None:
                    break

            try:
                depth, color = self._get_frames()
                if depth is None or color is None:
                    time.sleep(0.01)
                    continue

                # YOLO inference (object detection)
                results = self._model.predict(color, verbose=False)

                # Draw detections
                for r in results:
                    if r.boxes is None:
                        continue
                    for b in r.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
                        name = self._model.names.get(cls_id, str(cls_id))

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        dist_m = self._depth_at_center(depth, cx, cy)
                        dist_txt = f"{dist_m:.2f}m" if dist_m is not None else "?.??m"

                        label = f"{name} {conf:.2f} {dist_txt}"

                        # bbox
                        cv2.rectangle(color, (x1, y1), (x2, y2), (60, 180, 255), 2)

                        # label background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        ly = max(0, y1 - th - 8)
                        cv2.rectangle(color, (x1, ly), (x1 + tw + 10, ly + th + 8), (0, 0, 0), -1)
                        cv2.putText(color, label, (x1 + 5, ly + th + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                        # center sample box (optional)
                        k = max(1, int(CONFIG.depth_kernel))
                        off = k // 2
                        cv2.rectangle(color, (cx - off, cy - off), (cx + off, cy + off), (255, 255, 255), 1)

                ok, jpg = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG.mjpeg_quality)])
                if not ok:
                    continue

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                #time.sleep(1.0 / max(1, CONFIG.rs_fps))

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.03)
                continue
