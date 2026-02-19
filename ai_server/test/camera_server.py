import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import pyrealsense2 as rs
import numpy as np
import cv2
import threading

app = FastAPI(title="RealSense Eye-in-Hand API Server")

class CameraSystem:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable depth and color streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Align depth to color (Crucial for accurate clicking)
        self.align = rs.align(rs.stream.color)
        
        print("[INFO] Starting RealSense camera...")
        profile = self.pipeline.start(config)
        
        # Get lens intrinsics for 3D math
        depth_stream = profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        self.last_depth_frame = None
        self.lock = threading.Lock()

    def get_frames(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Safely store the latest depth frame for API queries
            with self.lock:
                self.last_depth_frame = depth_frame

            # Stream color image as MJPEG
            color_image = np.asanyarray(color_frame.get_data())
            ret, buffer = cv2.imencode('.jpg', color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def calculate_3d_point(self, pixel_x: int, pixel_y: int):
        with self.lock:
            if not self.last_depth_frame:
                return {"valid": False, "error": "Depth frame not ready yet"}
            
            # Read distance (in meters)
            dist = self.last_depth_frame.get_distance(pixel_x, pixel_y)
            if dist <= 0:
                return {"valid": False, "error": "Invalid distance (Too close/far or reflective surface)"}

            # Deproject 2D pixel to 3D point (Camera coordinate system)
            point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], dist)
            
            return {
                "valid": True,
                "x_m": point_3d[0],
                "y_m": point_3d[1],
                "z_m": point_3d[2],
                "distance_m": dist
            }

cam = CameraSystem()

class PixelQuery(BaseModel):
    x: int
    y: int

@app.get("/video_feed")
def video_feed():
    """Provides MJPEG stream to the Mac"""
    return StreamingResponse(cam.get_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/get_3d_coordinate")
def get_3d_coordinate(query: PixelQuery):
    """Calculates 3D real-world coordinates for a given pixel"""
    # Ignore phantom clicks at (0,0) that happen during window initialization
    if query.x == 0 and query.y == 0:
        return JSONResponse(status_code=400, content={"valid": False, "error": "Ignored origin (0,0)"})

    result = cam.calculate_3d_point(query.x, query.y)
    
    if not result["valid"]:
        return JSONResponse(status_code=400, content=result)
    
    return result

if __name__ == "__main__":
    print("[INFO] Server is running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)