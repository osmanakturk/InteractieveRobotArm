# ai_server/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class PickAndPlaceConfig:
    # Gateway base URL is passed via env: GATEWAY_URL
    stream_path: str = "/api/cameras/realsense/stream/mjpeg"
    grasp_pack_path: str = "/api/cameras/realsense/get_grasp_pack"

    # Required robot endpoints (absolute pose)
    robot_move_pose_path: str = "/api/robot/move_pose"
    robot_enable_path: str = "/api/robot/enable"
    robot_stop_path: str = "/api/robot/stop"
    robot_home_path: str = "/api/robot/home"
    gripper_status_path: str = "/api/robot/gripper/status"
    gripper_jog_path: str = "/api/robot/gripper"

    # Vision click/crop
    crop_size: int = 300
    stabilize_sleep_s: float = 0.12

    # Optional modes
    yolo_mode: bool = False
    debug_mode: bool = True
    reload_h_on_key: bool = True

    # Models
    yolo_model_name: str = "models/yolo26x-seg.pt"
    ggcnn_weight_path: str = "ggcnn_weights.pt"

    # Homography
    homography_file: str = "H_img2base.npy"

    # Measurements / safety
    table_z_mm: float = 183.195312
    clearance_mm: float = 20.0
    max_drop_mm: float = 220.0

    # Vision pose
    vision_x: float = 313.177277
    vision_y: float = 1.3344
    vision_z: float = 400.0
    vision_roll: float = -179.829107
    vision_pitch: float = 0.058213
    vision_yaw: float = -1.014021

    # Pick/Place
    pick_clearance_mm: float = 5.0
    place_clearance_mm: float = 25.0
    place_z_mm: Optional[float] = None  # if None => table_z + place_clearance

    pick_attempts: int = 3
    pick_step_down_mm: float = 2.0

    # Motion speeds
    speed_move: int = 120
    speed_descend: int = 28
    speed_ascend: int = 90

    # Gripper (targets; gateway is jog-based)
    gripper_open_pos: int = 850
    gripper_closed_pos: int = 0

    # Auto yaw
    enable_auto_yaw: bool = False
    yaw_sign: float = 1.0
    yaw_offset_deg: float = 0.0

    # Network
    http_timeout_s: float = 3.0


@dataclass(frozen=True)
class AiServerConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = True

    # Mode manager
    single_active_mode: bool = True
    health_ok_message: str = "alive"

    mode_entrypoints: Dict[str, str] = None  # type: ignore[assignment]
    pick_and_place: PickAndPlaceConfig = PickAndPlaceConfig()


def _default_entrypoints() -> Dict[str, str]:
    base = Path(__file__).resolve().parent
    return {
        "pick_and_place": str(base / "pick_and_place.py"),
        "voice_control": str(base / "voice_control.py"),
    }


CONFIG = AiServerConfig(mode_entrypoints=_default_entrypoints())