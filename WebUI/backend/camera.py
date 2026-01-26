# backend/camera.py
import time
from typing import Generator, Optional

import cv2

from config import CONFIG


class CameraStreamer:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self.cap is not None:
            return True
        cap = cv2.VideoCapture(CONFIG.camera_index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera_height)
        cap.set(cv2.CAP_PROP_FPS, CONFIG.camera_fps)
        self.cap = cap
        return True

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def frames(self) -> Generator[bytes, None, None]:
        """MJPEG generator."""
        if not self.open():
            # return a basic error frame
            import numpy as np
            img = np.zeros((CONFIG.camera_height, CONFIG.camera_width, 3), dtype=np.uint8)
            cv2.putText(img, "Camera not available", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, jpg = cv2.imencode(".jpg", img)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            return

        while True:
            assert self.cap is not None
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(1.0 / max(1, CONFIG.camera_fps))
