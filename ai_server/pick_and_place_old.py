# ai_server/pick_and_place.py

# Hotkeys:
#   q : quit
#   v : go vision pose
#   e : recover (enable + stop)
#   l : reload homography (if enabled)
#   o : open gripper
#   c : close gripper

from __future__ import annotations

import base64
import math
import os
import threading
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import cv2
import httpx
import numpy as np
import scipy.ndimage as ndimage
import torch
from skimage.draw import disk
from skimage.feature import peak_local_max

from config import CONFIG



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")


def normalize_any_http(input_value: str) -> str:
    raw = (input_value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw.rstrip("/")
    return f"http://{raw}".rstrip("/")


GATEWAY_URL = normalize_any_http(os.environ.get("GATEWAY_URL", ""))
PPC = CONFIG.pick_and_place

YOLO = None  # global default

if PPC.yolo_mode:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "yolo_mode=True but ultralytics is not installed. "
            "Install with: pip install ultralytics"
        ) from e
# =============================================================================
# GGCNN import (keep your structure)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import models.ggcnn2  # noqa: E402
from models.ggcnn2 import GGCNN2  # noqa: E402


# =============================================================================
# GLOBAL STATE
# =============================================================================
robot_state = "WAIT_TO_PICK"  # WAIT_TO_PICK | WAIT_TO_PLACE | MOVING
click_target: Optional[Tuple[int, int]] = None
last_click_pixel: Optional[Tuple[int, int]] = None

prev_mp = np.array([150, 150], dtype=np.int64)
last_debug_img = np.zeros((300, 300, 3), dtype=np.uint8)


# =============================================================================
# Gateway API Client
# =============================================================================
class GatewayApi:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _url(self, path: str) -> str:
        return self.base + path

    def status(self) -> Dict[str, Any]:
        r = self.client.get(self._url("/api/status"))
        r.raise_for_status()
        return r.json()

    # ---- Camera
    def stream_url(self) -> str:
        return self._url(PPC.stream_path)

    def ensure_camera_started(self) -> None:
        try:
            st = self.status()
            started = bool(st.get("cameras", {}).get("realsense", {}).get("started"))
            if not started:
                self.client.post(self._url("/api/cameras/realsense/start"))
        except Exception:
            pass

    def get_grasp_pack(self, x: int, y: int, crop_size: int) -> Dict[str, Any]:
        payload = {"x": int(x), "y": int(y), "crop_size": int(crop_size)}
        r = self.client.post(self._url(PPC.grasp_pack_path), json=payload)
        if r.status_code != 200:
            return {"valid": False, "message": f"gateway grasp_pack http={r.status_code}", "raw": _safe_json(r)}
        return r.json()

    # ---- Robot
    def robot_enable(self) -> None:
        self.client.post(self._url(PPC.robot_enable_path))

    def robot_stop(self) -> None:
        self.client.post(self._url(PPC.robot_stop_path))

    def robot_home(self) -> None:
        self.client.post(self._url(PPC.robot_home_path))

    def robot_move_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        speed: int,
        wait: bool = True,
    ) -> Tuple[bool, str]:
        body = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "speed": int(speed),
            "wait": bool(wait),
        }
        path = PPC.robot_move_pose_path
        try:
            r = self.client.post(self._url(path), json=body)
            if r.status_code == 200:
                return True, "OK"
            return False, f"{path} http={r.status_code} {_safe_json(r)}"
        except Exception as e:
            return False, f"{path} error: {e}"

    # ---- Gripper
    def gripper_status(self) -> Dict[str, Any]:
        r = self.client.get(self._url(PPC.gripper_status_path))
        if r.status_code != 200:
            return {"available": False}
        data = r.json()
        return data.get("gripper", {}) if isinstance(data, dict) else {"available": False}

    def gripper_jog(self, action: str) -> None:
        self.client.post(self._url(PPC.gripper_jog_path), json={"action": action})


def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return {"text": r.text[:400]}


# =============================================================================
# UTILS
# =============================================================================
def safe_z(z: float) -> float:
    min_z = float(PPC.table_z_mm + PPC.clearance_mm)
    return max(min_z, float(z))


def normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def load_homography() -> Optional[np.ndarray]:
    f = PPC.homography_file
    if os.path.exists(f):
        try:
            H = np.load(f)
            if H.shape == (3, 3):
                return H
        except Exception:
            return None
    return None


def project_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def recover_robot(gw: GatewayApi) -> None:
    try:
        gw.robot_stop()
    except Exception:
        pass
    try:
        gw.robot_enable()
    except Exception:
        pass


def go_vision_pose(gw: GatewayApi) -> None:
    try:
        gw.robot_enable()
    except Exception:
        pass
    ok, msg = gw.robot_move_pose(
        x=PPC.vision_x,
        y=PPC.vision_y,
        z=PPC.vision_z,
        roll=PPC.vision_roll,
        pitch=PPC.vision_pitch,
        yaw=PPC.vision_yaw,
        speed=PPC.speed_move,
        wait=True,
    )
    if not ok:
        raise RuntimeError(msg)


def ensure_gripper_position(gw: GatewayApi, target: int) -> None:
    """
    Gateway jog open/close until close enough (if position is readable).
    """
    for _ in range(int(PPC.gripper_max_steps)):
        st = gw.gripper_status()
        if not st.get("available"):
            # no feedback; do one jog and stop
            gw.gripper_jog("close" if target <= PPC.gripper_closed_pos else "open")
            time.sleep(float(PPC.gripper_step_sleep_s))
            return

        pos = st.get("pos")
        if pos is None:
            gw.gripper_jog("close" if target <= PPC.gripper_closed_pos else "open")
            time.sleep(float(PPC.gripper_step_sleep_s))
            continue

        pos_i = int(pos)
        if abs(pos_i - int(target)) <= int(PPC.gripper_tolerance):
            return

        gw.gripper_jog("open" if pos_i < target else "close")
        time.sleep(float(PPC.gripper_step_sleep_s))


def is_grasp_likely(gw: GatewayApi) -> Optional[bool]:
    """
    Heuristic:
      - After closing, if gripper can't fully reach closed_pos and stays more open
        than (closed_pos + grasped_pos_delta), likely object is between fingers.
    Returns None if no feedback.
    """
    st = gw.gripper_status()
    if not st.get("available"):
        return None
    pos = st.get("pos")
    if pos is None:
        return None
    pos_i = int(pos)
    return pos_i >= int(PPC.gripper_closed_pos + PPC.grasped_pos_delta)


# =============================================================================
# GGCNN
# =============================================================================
def process_depth_image(depth_m: np.ndarray, out_size: int = 300):
    depth = depth_m.copy()
    invalid = (depth <= 0.0).astype(np.uint8)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    invalid = cv2.copyMakeBorder(invalid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

    kernel = np.ones((3, 3), np.uint8)
    invalid = cv2.dilate(invalid, kernel, iterations=1)
    depth[invalid == 1] = 0.0

    scale = float(np.max(depth)) if float(np.max(depth)) > 0 else 1.0
    depth_norm = (depth / scale).astype(np.float32)

    depth_inp = cv2.inpaint(depth_norm, invalid, 1, cv2.INPAINT_NS) * scale
    depth_inp = depth_inp[1:-1, 1:-1]
    invalid = invalid[1:-1, 1:-1]

    depth_inp = cv2.resize(depth_inp, (out_size, out_size), cv2.INTER_AREA)
    invalid = cv2.resize(invalid, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_inp, invalid


def load_ggcnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(SCRIPT_DIR, PPC.ggcnn_weight_path)

    torch.serialization.add_safe_globals([GGCNN2])
    model = torch.load(weight_path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model, device


def ggcnn_infer(depth_crop_m: np.ndarray, model, device, out_size: int):
    global prev_mp

    depth_proc, invalid_mask = process_depth_image(depth_crop_m, out_size)
    depth_norm = np.clip((depth_proc - depth_proc.mean()), -1, 1)
    depthT = torch.from_numpy(depth_norm.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)

    with torch.no_grad():
        pred = model(depthT)

    points = pred[0].cpu().numpy().squeeze()
    cos_o = pred[1].cpu().numpy().squeeze()
    sin_o = pred[2].cpu().numpy().squeeze()
    ang = np.arctan2(sin_o, cos_o) / 2.0

    points[invalid_mask == 1] = 0
    points = ndimage.gaussian_filter(points, 5.0)
    ang = ndimage.gaussian_filter(ang, 2.0)
    points = np.clip(points, 0.0, 1.0 - 1e-3)

    maxes = peak_local_max(points, min_distance=10, threshold_abs=0.1, num_peaks=3)
    if maxes.shape[0] == 0:
        return None

    mp = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]
    if np.linalg.norm(mp - prev_mp) > 30:
        mp = np.array(np.unravel_index(np.argmax(points), points.shape))
        prev_mp = mp.astype(np.int64)
    else:
        prev_mp = (mp * 0.25 + prev_mp * 0.75).astype(np.int64)

    angle_rad = float(ang[prev_mp[0], prev_mp[1]])
    angle_deg = math.degrees(angle_rad)

    debug_img = None
    if PPC.debug_mode:
        depth_vis = cv2.normalize(depth_proc, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.applyColorMap((points * 255).astype(np.uint8), cv2.COLORMAP_JET)
        debug_img = cv2.addWeighted(depth_vis, 0.4, heatmap, 0.6, 0)

        rr, cc = disk(prev_mp, 5, shape=(out_size, out_size, 3))
        debug_img[rr, cc] = (0, 255, 0)

        L = 25
        dx = int(L * math.cos(angle_rad))
        dy = int(L * math.sin(angle_rad))
        cv2.line(
            debug_img,
            (prev_mp[1] - dx, prev_mp[0] - dy),
            (prev_mp[1] + dx, prev_mp[0] + dy),
            (0, 0, 255),
            3,
        )

    return {
        "best_u": int(prev_mp[1]),
        "best_v": int(prev_mp[0]),
        "angle_deg": float(angle_deg),
        "debug_img": debug_img,
    }


# =============================================================================
# INPUT
# =============================================================================
def mouse_callback(event, x, y, flags, param):
    global click_target, last_click_pixel, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        click_target = (x, y)
        last_click_pixel = (x, y)


# =============================================================================
# ROBOT ACTION (via gateway)
# =============================================================================
def perform_pick_or_place(gw: GatewayApi, x: float, y: float, z: float, yaw_deg: float, action: str):
    """
    Guarantees: if configured, always returns to vision pose even on cancel/error.
    """
    global robot_state
    cancelled = False
    try:
        robot_state = "MOVING"
        recover_robot(gw)

        # Always clamp
        z = safe_z(z)

        # Move XY at vision_z
        ok, _ = gw.robot_move_pose(
            x=x, y=y, z=PPC.vision_z,
            roll=PPC.vision_roll, pitch=PPC.vision_pitch, yaw=yaw_deg,
            speed=PPC.speed_move, wait=True
        )
        if not ok:
            cancelled = True
            robot_state = "WAIT_TO_PICK"
            return

        if PPC.dry_run_xy_only:
            # Only XY move, no descend
            robot_state = "WAIT_TO_PLACE" if action == "PICK" else "WAIT_TO_PICK"
            return

        drop = float(PPC.vision_z) - float(z)
        if drop > float(PPC.max_drop_mm):
            cancelled = True
            robot_state = "WAIT_TO_PLACE" if action == "PICK" else "WAIT_TO_PICK"
            return

        # Descend
        ok, _ = gw.robot_move_pose(
            x=x, y=y, z=z,
            roll=PPC.vision_roll, pitch=PPC.vision_pitch, yaw=yaw_deg,
            speed=PPC.speed_descend, wait=True
        )
        if not ok:
            cancelled = True
            robot_state = "WAIT_TO_PICK"
            return

        if action == "PICK":
            ensure_gripper_position(gw, PPC.gripper_closed_pos)
            time.sleep(0.2)
            # Ascend back
            gw.robot_move_pose(
                x=x, y=y, z=PPC.vision_z,
                roll=PPC.vision_roll, pitch=PPC.vision_pitch, yaw=yaw_deg,
                speed=PPC.speed_ascend, wait=True
            )
            robot_state = "WAIT_TO_PLACE"
        else:
            ensure_gripper_position(gw, PPC.gripper_open_pos)
            time.sleep(0.2)
            gw.robot_move_pose(
                x=x, y=y, z=PPC.vision_z,
                roll=PPC.vision_roll, pitch=PPC.vision_pitch, yaw=yaw_deg,
                speed=PPC.speed_ascend, wait=True
            )
            robot_state = "WAIT_TO_PICK"

    except Exception:
        cancelled = True
        robot_state = "WAIT_TO_PICK"
    finally:
        # This is the key fix: ALWAYS return to standard pose
        try:
            if cancelled and not PPC.return_to_vision_pose_on_cancel:
                return
            if (not cancelled) and not PPC.return_to_vision_pose_after_action:
                return
            if cancelled and PPC.return_to_vision_pose_on_error:
                go_vision_pose(gw)
            elif not cancelled and PPC.return_to_vision_pose_after_action:
                go_vision_pose(gw)
        except Exception:
            pass


def pick_attempts_thread(gw: GatewayApi, tx: float, ty: float, yaw_target: float, pick_z_base: float):
    """
    Multiple attempts:
      - step down
      - close gripper
      - (optional) verify grasp via gripper position heuristic
      - if not grasped: open and try lower
    """
    global robot_state

    for attempt in range(int(PPC.pick_attempts)):
        tz = safe_z(pick_z_base - float(attempt) * float(PPC.pick_step_down_mm))

        perform_pick_or_place(gw, tx, ty, tz, yaw_target, "PICK")

        if not PPC.verify_grasp_via_gripper:
            return

        # Verify after PICK: if robot_state is WAIT_TO_PLACE, check if grasp likely.
        if robot_state != "WAIT_TO_PLACE":
            # something went wrong; stop attempts
            return

        likely = is_grasp_likely(gw)
        if likely is None:
            # no feedback; accept
            return

        if likely:
            # grasp ok
            return

        # Not grasped -> open and retry lower
        ensure_gripper_position(gw, PPC.gripper_open_pos)
        time.sleep(0.15)
        robot_state = "WAIT_TO_PICK"

    # attempts done; ensure state is consistent
    robot_state = "WAIT_TO_PICK"


# =============================================================================
# MAIN
# =============================================================================
def main():
    global click_target, robot_state, last_debug_img

    if not GATEWAY_URL:
        raise RuntimeError("GATEWAY_URL is empty. Start this mode via ai_server/main.py with gateway_url.")

    gw = GatewayApi(GATEWAY_URL, timeout_s=float(PPC.http_timeout_s))
    gw.ensure_camera_started()

    ggcnn_model, device = load_ggcnn_model()
    H = load_homography()

    yolo = None
    if PPC.yolo_mode:
        if YOLO is None:
            raise RuntimeError("yolo_mode=True but ultralytics is not available.")
        yolo = YOLO(PPC.yolo_model_name)

    # Go standard pose first
    go_vision_pose(gw)

    stream_url = gw.stream_url()
    cap = cv2.VideoCapture(stream_url)

    cv2.namedWindow(PPC.window_name)
    cv2.setMouseCallback(PPC.window_name, mouse_callback)
    if PPC.debug_mode:
        cv2.namedWindow(PPC.debug_window_name)

    pick_z_base = float(PPC.table_z_mm + PPC.pick_clearance_mm)
    place_z = float(PPC.place_z_mm) if PPC.place_z_mm is not None else float(PPC.table_z_mm + PPC.place_clearance_mm)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.25)
            cap.open(stream_url)
            continue

        show = frame
        results = None
        if PPC.yolo_mode and yolo is not None:
            results = yolo(frame, verbose=False)

        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None

            # Stabilize view
            go_vision_pose(gw)
            time.sleep(float(PPC.stabilize_sleep_s))

            # YOLO validation (PICK only) -- FIXED: invalid => continue
            if PPC.yolo_mode and robot_state == "WAIT_TO_PICK":
                valid = False
                if results is None:
                    results = yolo(frame, verbose=False)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            valid = True
                            break
                if not valid:
                    continue

            try:
                pack = gw.get_grasp_pack(int(cx), int(cy), int(PPC.crop_size))
                if not pack.get("valid"):
                    continue

                depth_scale = float(pack["depth_scale"])
                origin = pack["crop_origin"]

                depth_png = base64.b64decode(pack["depth_crop_b64"])
                depth_u16 = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_ANYDEPTH)
                if depth_u16 is None:
                    continue
                depth_crop_m = depth_u16.astype(np.float32) * depth_scale

                g = ggcnn_infer(depth_crop_m, ggcnn_model, device, out_size=int(PPC.crop_size))
                if g is None:
                    continue
                if g["debug_img"] is not None:
                    last_debug_img = g["debug_img"]

                crop_h, crop_w = depth_u16.shape[:2]
                u_in_crop = int(g["best_u"] / float(PPC.crop_size) * crop_w)
                v_in_crop = int(g["best_v"] / float(PPC.crop_size) * crop_h)

                u_full = int(origin["x"] + u_in_crop)
                v_full = int(origin["y"] + v_in_crop)

                if H is None:
                    H = load_homography()
                if H is None:
                    continue

                tx, ty = project_xy(H, u_full, v_full)

                action = "PICK" if robot_state == "WAIT_TO_PICK" else "PLACE"

                yaw_target = float(PPC.vision_yaw)
                if action == "PICK" and PPC.enable_auto_yaw:
                    yaw_target = normalize_deg(
                        float(PPC.vision_yaw) + float(PPC.yaw_sign) * float(g["angle_deg"]) + float(PPC.yaw_offset_deg)
                    )

                if action == "PLACE":
                    th = threading.Thread(
                        target=perform_pick_or_place,
                        args=(gw, tx, ty, place_z, yaw_target, "PLACE"),
                        daemon=True,
                    )
                    th.start()
                else:
                    th = threading.Thread(
                        target=pick_attempts_thread,
                        args=(gw, tx, ty, yaw_target, pick_z_base),
                        daemon=True,
                    )
                    th.start()

            except Exception:
                recover_robot(gw)
                robot_state = "WAIT_TO_PICK"

        cv2.imshow(PPC.window_name, show)
        if PPC.debug_mode:
            cv2.imshow(PPC.debug_window_name, last_debug_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            go_vision_pose(gw)
        if key == ord("e"):
            recover_robot(gw)
        if key == ord("l") and PPC.reload_h_on_key:
            H = load_homography()
        if key == ord("o"):
            ensure_gripper_position(gw, PPC.gripper_open_pos)
        if key == ord("c"):
            ensure_gripper_position(gw, PPC.gripper_closed_pos)

    cap.release()
    cv2.destroyAllWindows()
    try:
        go_vision_pose(gw)
    except Exception:
        pass
    gw.close()


if __name__ == "__main__":
    main()