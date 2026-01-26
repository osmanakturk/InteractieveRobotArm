# backend/config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    # UI / backend
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    # Robot defaults
    default_ip: str = ""
    # Jog settings
    default_speed_pct: int = 50   # 1..100
    default_xyz_step_mm: float = 5.0
    default_rpy_step_deg: float = 2.0

    # Gripper (typical xArm gripper range: 0..850 or 0..1000 depending on model)
    gripper_min: int = 0
    gripper_max: int = 850
    gripper_step: int = 30

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 360
    camera_fps: int = 30

CONFIG = AppConfig()
