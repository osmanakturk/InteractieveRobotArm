# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Robot
    default_speed_pct: int = 50
    gripper_min: int = 0
    gripper_max: int = 850
    gripper_step: int = 30

    # RealSense
    rs_width: int = 640
    rs_height: int = 480
    rs_fps: int = 30

    # Streaming (MJPEG)
    mjpeg_quality: int = 80

    # YOLO model (segmentation)
    yolo_model_path: str = "models/yolo26n-seg.engine"

    # Depth label sampling (bbox center 5x5 average)
    depth_kernel: int = 5  # must be odd

    # Optional: overlay alpha
    overlay_alpha: float = 0.4  # mask fill alpha

CONFIG = AppConfig()
