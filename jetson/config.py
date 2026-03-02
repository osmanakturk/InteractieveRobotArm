# jetson/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


# ----------------
# Nested configs
# ----------------
@dataclass(frozen=True)
class VisionPose:
    # Units: mm + degrees
    x: float = 250.0
    y: float = 0.0
    z: float = 350.0
    roll: float = 180.0
    pitch: float = 0.0
    yaw: float = 0.0
    speed: int = 120
    wait: bool = True


@dataclass(frozen=True)
class RobotConfig:
    # The reference pose used by vision/pick-place pipelines
    vision_pose: VisionPose = field(default_factory=VisionPose)


# ----------------
# App config
# ----------------
@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ----------------
    # Robot (flat fields kept for backward compatibility)
    # ----------------
    default_speed_pct: int = 50
    gripper_min: int = 0
    gripper_max: int = 850
    gripper_step: int = 30

    # If True: after connect(), gateway will auto-enable robot (motion_enable)
    robot_autostart_enable: bool = True

    # NEW: nested robot config (CONFIG.robot.vision_pose)
    robot: RobotConfig = field(default_factory=RobotConfig)

    # Optional: if later you confirm xArm state codes that mean "moving",
    # you can fill these values. Leave empty for now.
    robot_state_moving_values: Tuple[int, ...] = ()

    # ----------------
    # SAFETY LIMITS (None => no limit)
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
    # ----------------
    jog_min_interval_ms: int = 100

    # ----------------
    # RealSense
    # ----------------
    rs_width: int = 640
    rs_height: int = 480
    rs_fps: int = 30

    mjpeg_quality: int = 80

    # If True: streamer opens automatically on init
    camera_autostart: bool = True


    @property
    def robot_vision_pose(self) -> Dict[str, Any]:
        vp = self.robot.vision_pose
        return {
            "x": vp.x,
            "y": vp.y,
            "z": vp.z,
            "roll": vp.roll,
            "pitch": vp.pitch,
            "yaw": vp.yaw,
            "speed": vp.speed,
            "wait": vp.wait,
        }


CONFIG = AppConfig()