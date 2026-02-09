# jetson/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ----------------
    # Robot
    # ----------------
    default_speed_pct: int = 50
    gripper_min: int = 0
    gripper_max: int = 850
    gripper_step: int = 30

    # ----------------
    # SAFETY LIMITS (None => no limit)
    # Only set what you need; others stay unlimited.
    # ----------------
    safety_x_min_mm: Optional[float] = None
    safety_x_max_mm: Optional[float] = None
    safety_y_min_mm: Optional[float] = None
    safety_y_max_mm: Optional[float] = None
    safety_z_min_mm: Optional[float] = None
    safety_z_max_mm: Optional[float] = None

    safety_roll_min_deg: Optional[float] = None
    safety_roll_max_deg: Optional[float] = None
    safety_pitch_min_deg: Optional[float] = None
    safety_pitch_max_deg: Optional[float] = None
    safety_yaw_min_deg: Optional[float] = None
    safety_yaw_max_deg: Optional[float] = None

    # ----------------
    # Rate limiting / jog throttle (server-side)
    # If /api/jog comes faster than this, return 429.
    # Recommended: 80..120ms
    # ----------------
    jog_min_interval_ms: int = 100

    # ----------------
    # RealSense
    # ----------------
    rs_width: int = 640
    rs_height: int = 480
    rs_fps: int = 30

    mjpeg_quality: int = 80
    yolo_model_path: str = "models/yolo26n-seg.engine"

    depth_kernel: int = 5  # must be odd
    overlay_alpha: float = 0.4

    camera_autostart: bool = False


CONFIG = AppConfig()
