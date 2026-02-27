import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import base64
import time

app = FastAPI(title="RealSense Eye-in-Hand Camera Server")

class PixelQuery(BaseModel):
    x: int
    y: int
    crop_size: int = 300

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class RealSenseServer:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.align = rs.align(rs.stream.color)

        print("[INFO] Starting RealSense...")
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"[INFO] depth_scale={self.depth_scale} m/unit")

        # IMPORTANT: because we align depth->color, use COLOR intrinsics
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intr = color_stream.get_intrinsics()
        self.w = int(self.color_intr.width)
        self.h = int(self.color_intr.height)

        self.lock = threading.Lock()
        self.last_depth_aligned = None   # uint16 aligned to color
        self.last_color = None

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            depth_np = np.asanyarray(depth.get_data())  # uint16
            color_np = np.asanyarray(color.get_data())  # bgr8

            with self.lock:
                self.last_depth_aligned = depth_np
                self.last_color = color_np

    def mjpeg_generator(self, jpeg_quality=80):
        while True:
            with self.lock:
                img = None if self.last_color is None else self.last_color.copy()
            if img is None:
                time.sleep(0.02)
                continue

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    def get_grasp_pack(self, cx: int, cy: int, crop_size: int):
        with self.lock:
            if self.last_depth_aligned is None:
                return {"valid": False, "error": "Depth not ready"}
            depth = self.last_depth_aligned

        cx = int(clamp(cx, 0, self.w - 1))
        cy = int(clamp(cy, 0, self.h - 1))
        crop_size = int(clamp(crop_size, 64, 640))
        half = crop_size // 2

        x1 = clamp(cx - half, 0, self.w - 1)
        y1 = clamp(cy - half, 0, self.h - 1)
        x2 = clamp(cx + half, 0, self.w)
        y2 = clamp(cy + half, 0, self.h)

        depth_crop = depth[y1:y2, x1:x2]  # uint16 aligned-to-color

        ok, png = cv2.imencode(".png", depth_crop)
        if not ok:
            return {"valid": False, "error": "PNG encode failed"}
        depth_b64 = base64.b64encode(png).decode("utf-8")

        intr = self.color_intr
        return {
            "valid": True,
            "depth_crop_b64": depth_b64,
            "crop_origin": {"x": int(x1), "y": int(y1)},
            "crop_size": {"w": int(depth_crop.shape[1]), "h": int(depth_crop.shape[0])},
            "depth_scale": self.depth_scale,
            "intrinsics": {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
            },
        }

server = RealSenseServer()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(server.mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/get_grasp_pack")
def api_get_grasp_pack(q: PixelQuery):
    if q.x == 0 and q.y == 0:
        return JSONResponse(status_code=400, content={"valid": False, "error": "Ignored origin"})
    pack = server.get_grasp_pack(q.x, q.y, q.crop_size)
    if not pack["valid"]:
        return JSONResponse(status_code=400, content=pack)
    return pack

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)