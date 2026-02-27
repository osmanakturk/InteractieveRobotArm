import os
import time
import math
import base64
import threading
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from skimage.draw import disk
from ultralytics import YOLO
from xarm.wrapper import XArmAPI

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")

# =========================
# CONFIG (EDIT THESE)
# =========================
CAMERA_SERVER_IP = "10.2.172.118"
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_grasp_pack"

ROBOT_IP = "10.2.172.20"

YOLO_MODEL_NAME = "models/yolo26x-seg.pt"
GGCNN_WEIGHT_PATH = "ggcnn_weights.pt"

DEBUG_MODE = True

# --- SAFETY Z from your measurements ---
TABLE_Z_MM = 183.195312
CLEARANCE_MM = 40.0
MIN_Z = TABLE_Z_MM + CLEARANCE_MM            # 223.195312

HOME_X = 313.177277
HOME_Y = 1.3344
HOME_Z = 251.968369

# Recommended safe hover: above MIN_Z + 120mm
HOVER_Z = max(HOME_Z, MIN_Z + 120.0)         # 343.195312

# Avoid huge drops from hover
MAX_DROP_MM = 140.0

# Gripper
GRIPPER_OPEN = 850
GRIPPER_CLOSED = 0

# If you don't have real hand-eye calibration yet:
USE_HAND_EYE_Z = False  # keep False for safety (no depth-based Z landing)

# =========================
# HAND-EYE (PLACEHOLDER!)
# Camera frame -> Tool frame transform (mm)
# Replace with your real calibration output when you have it.
# =========================
T_TOOL_CAM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# =========================
# GGCNN import
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import models.ggcnn2
from models.ggcnn2 import GGCNN2

prev_mp = np.array([150, 150], dtype=np.int64)
last_debug_img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(last_debug_img, "WAITING...", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# =========================
# Math / Transform helpers
# =========================
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def pose_to_T_base_tool(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    R = rot_z(y) @ rot_y(p) @ rot_x(r)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=np.float64)
    return T

def transform_point(T, p3):
    ph = np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)
    out = T @ ph
    return out[:3]

def deproject_pixel_to_cam(u, v, depth_m, intr):
    fx, fy, ppx, ppy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    X = (u - ppx) / fx * depth_m
    Y = (v - ppy) / fy * depth_m
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)  # meters

def camera_angle_to_tool_yaw(angle_rad_cam, T_tool_cam):
    R_tool_cam = T_tool_cam[:3, :3]
    v_cam = np.array([math.cos(angle_rad_cam), math.sin(angle_rad_cam), 0.0], dtype=np.float64)
    v_tool = R_tool_cam @ v_cam
    yaw_tool = math.atan2(v_tool[1], v_tool[0])
    return math.degrees(yaw_tool)

# =========================
# GGCNN processing
# =========================
def process_depth_image(depth_m: np.ndarray, out_size=300):
    depth = depth_m.copy()
    invalid = (depth <= 0.0).astype(np.uint8)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    invalid = cv2.copyMakeBorder(invalid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

    kernel = np.ones((3, 3), np.uint8)
    invalid = cv2.dilate(invalid, kernel, iterations=1)
    depth[invalid == 1] = 0.0

    scale = float(np.max(depth)) if float(np.max(depth)) > 0 else 1.0
    depth_norm = (depth / scale).astype(np.float32)

    depth_inp = cv2.inpaint(depth_norm, invalid, 1, cv2.INPAINT_NS)
    depth_inp = depth_inp * scale

    depth_inp = depth_inp[1:-1, 1:-1]
    invalid = invalid[1:-1, 1:-1]

    depth_inp = cv2.resize(depth_inp, (out_size, out_size), cv2.INTER_AREA)
    invalid = cv2.resize(invalid, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_inp, invalid

def load_ggcnn_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Loading GGCNN2 on {device}...")
    weight_path = os.path.join(SCRIPT_DIR, GGCNN_WEIGHT_PATH)
    torch.serialization.add_safe_globals([models.ggcnn2.GGCNN2])
    model = torch.load(weight_path, map_location=device, weights_only=False)
    model.to(device).eval()
    print("[SUCCESS] GGCNN2 loaded.")
    return model, device

def ggcnn_infer(depth_crop_m: np.ndarray, model, device, out_size=300):
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
    if DEBUG_MODE:
        depth_vis = cv2.normalize(depth_proc, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.applyColorMap((points * 255).astype(np.uint8), cv2.COLORMAP_JET)
        debug_img = cv2.addWeighted(depth_vis, 0.4, heatmap, 0.6, 0)

        rr, cc = disk(prev_mp, 5, shape=(out_size, out_size, 3))
        debug_img[rr, cc] = (0, 255, 0)

        L = 25
        dx = int(L * math.cos(angle_rad))
        dy = int(L * math.sin(angle_rad))
        cv2.line(debug_img, (prev_mp[1] - dx, prev_mp[0] - dy), (prev_mp[1] + dx, prev_mp[0] + dy), (0, 0, 255), 3)
        cv2.putText(debug_img, f"Angle: {angle_deg:.1f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return {
        "best_u_300": int(prev_mp[1]),
        "best_v_300": int(prev_mp[0]),
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "debug_img": debug_img,
    }

# =========================
# Robot helpers
# =========================
def recover_if_error(arm: XArmAPI) -> bool:
    try:
        code, ew = arm.get_err_warn_code()
        if isinstance(ew, list) and len(ew) > 0 and ew[0] != 0:
            print(f"[RECOVER] err={ew[0]} warn={ew[1] if len(ew) > 1 else 0}")
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_state(0)
            time.sleep(0.2)
            return True
    except Exception:
        pass
    return False

def connect_robot() -> XArmAPI:
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    arm.set_gripper_enable(True)
    arm.set_gripper_position(GRIPPER_OPEN, wait=True)
    return arm

def go_home(arm: XArmAPI):
    recover_if_error(arm)
    arm.set_position(x=HOME_X, y=HOME_Y, z=HOVER_Z, roll=180, pitch=0, yaw=0, speed=120, wait=True)

# =========================
# Click + state
# =========================
robot_state = "WAIT_TO_PICK"   # WAIT_TO_PICK -> MOVING -> WAIT_TO_PLACE -> MOVING -> WAIT_TO_PICK
click_target: Optional[Tuple[int, int]] = None

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        click_target = (x, y)

def safe_target_z(requested_z: float) -> float:
    # Always keep above MIN_Z
    tz = max(MIN_Z, float(requested_z))
    # Never higher than hover (for safety, predictable)
    tz = min(tz, float(HOVER_Z))
    return tz

def do_motion_pick_or_place(arm: XArmAPI, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float, action: str):
    global robot_state
    try:
        robot_state = "MOVING"
        recover_if_error(arm)

        z_mm = safe_target_z(z_mm)

        # Approach from safe hover
        arm.set_position(x=x_mm, y=y_mm, z=HOVER_Z, roll=180, pitch=0, yaw=yaw_deg, speed=120, wait=True)

        # Prevent big drops
        drop = float(HOVER_Z - z_mm)
        if drop > MAX_DROP_MM:
            print(f"[SAFETY] Drop too large ({drop:.1f}mm) -> CANCEL")
            robot_state = "WAIT_TO_PICK" if action == "PICK" else "WAIT_TO_PLACE"
            return

        # Descend
        arm.set_position(z=z_mm, speed=60, wait=True)

        if action == "PICK":
            arm.set_gripper_position(GRIPPER_CLOSED, wait=True)
            time.sleep(0.25)
            arm.set_position(z=HOVER_Z, speed=90, wait=True)
            go_home(arm)
            robot_state = "WAIT_TO_PLACE"
        else:
            arm.set_gripper_position(GRIPPER_OPEN, wait=True)
            time.sleep(0.25)
            arm.set_position(z=HOVER_Z, speed=90, wait=True)
            go_home(arm)
            robot_state = "WAIT_TO_PICK"

    except Exception as e:
        print(f"[ERROR] Motion failed: {e}")
        # Try recover
        recover_if_error(arm)
        robot_state = "WAIT_TO_PICK"

# =========================
# Main pipeline
# =========================
def main():
    global click_target, robot_state, last_debug_img

    print("==================================================")
    print("✅ SAFE mac_controller (Eye-in-hand architecture)")
    print("==================================================")
    print(f"[SAFE] TABLE_Z={TABLE_Z_MM:.3f}  MIN_Z={MIN_Z:.3f}  HOVER_Z={HOVER_Z:.3f}")
    print(f"[SAFE] USE_HAND_EYE_Z={USE_HAND_EYE_Z} (False => fixed safe Z)")

    ggcnn_model, device = load_ggcnn_model()

    print(f"[INFO] Loading YOLO: {YOLO_MODEL_NAME}")
    yolo = YOLO(YOLO_MODEL_NAME)

    arm = connect_robot()
    go_home(arm)

    cap = cv2.VideoCapture(STREAM_URL)
    cv2.namedWindow("YOLO + GGCNN (SAFE)")
    cv2.setMouseCallback("YOLO + GGCNN (SAFE)", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            cap.open(STREAM_URL)
            continue

        results = yolo(frame, verbose=False)
        annotated = results[0].plot()

        # UI
        if robot_state == "WAIT_TO_PICK":
            cv2.putText(annotated, "CLICK OBJECT TO PICK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        elif robot_state == "WAIT_TO_PLACE":
            cv2.putText(annotated, "CLICK LOCATION TO PLACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        else:
            cv2.putText(annotated, "ROBOT MOVING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Click processing
        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None

            # Validate click inside YOLO bbox only when picking
            if robot_state == "WAIT_TO_PICK":
                valid = False
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            valid = True
                            break
                if not valid:
                    print("[WARN] Click not on an object. Cancel.")
                    continue

            try:
                # 1) request pack from camera server
                resp = requests.post(API_URL, json={"x": int(cx), "y": int(cy), "crop_size": 300}, timeout=3)
                if resp.status_code != 200:
                    print("[ERROR] Server:", resp.status_code, resp.text)
                    continue
                pack = resp.json()
                if not pack.get("valid"):
                    print("[ERROR] Pack invalid:", pack)
                    continue

                intr = pack["intrinsics"]
                depth_scale = float(pack["depth_scale"])
                origin = pack["crop_origin"]

                depth_png = base64.b64decode(pack["depth_crop_b64"])
                depth_u16 = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_ANYDEPTH)
                if depth_u16 is None:
                    print("[ERROR] Depth decode failed")
                    continue

                depth_crop_m = depth_u16.astype(np.float32) * depth_scale

                # 2) GGCNN -> best grasp point + angle
                g = ggcnn_infer(depth_crop_m, ggcnn_model, device, out_size=300)
                if g is None:
                    print("[ERROR] GGCNN: no grasp found")
                    continue
                if g["debug_img"] is not None:
                    last_debug_img = g["debug_img"]

                # map from 300x300 to original crop size
                crop_h, crop_w = depth_u16.shape[:2]
                u_in_crop = int(g["best_u_300"] / 300.0 * crop_w)
                v_in_crop = int(g["best_v_300"] / 300.0 * crop_h)

                u_full = int(origin["x"] + u_in_crop)
                v_full = int(origin["y"] + v_in_crop)

                depth_m = float(depth_crop_m[v_in_crop, u_in_crop])
                if depth_m <= 0:
                    print("[ERROR] Depth at grasp pixel invalid")
                    continue

                # 3) Deproject -> P_cam (meters)
                P_cam_m = deproject_pixel_to_cam(u_full, v_full, depth_m, intr)
                P_cam_mm = P_cam_m * 1000.0

                # 4) Read robot pose -> T_base_tool
                code, pose = arm.get_position(is_radian=False)
                if code != 0:
                    print("[ERROR] get_position failed")
                    continue
                x_mm, y_mm, z_mm, roll, pitch, yaw = pose
                T_base_tool = pose_to_T_base_tool(x_mm, y_mm, z_mm, roll, pitch, yaw)

                # 5) Eye-in-hand point transform:
                P_tool_mm = transform_point(T_TOOL_CAM, P_cam_mm)
                P_base_mm = transform_point(T_base_tool, P_tool_mm)

                # 6) Yaw handling (safe approx)
                yaw_tool_deg = camera_angle_to_tool_yaw(g["angle_rad"], T_TOOL_CAM)
                target_yaw_deg = float(yaw + yaw_tool_deg)

                # 7) SAFE target Z
                # - Without calibration: do NOT trust P_base_mm[2] for landing
                if USE_HAND_EYE_Z:
                    requested_z = float(P_base_mm[2])
                else:
                    requested_z = float(MIN_Z)

                target_x = float(P_base_mm[0])
                target_y = float(P_base_mm[1])
                target_z = safe_target_z(requested_z)

                print(f"[TARGET] u_full={u_full} v_full={v_full} depth={depth_m*1000:.1f}mm")
                print(f"[TARGET] base(mm)=({target_x:.1f},{target_y:.1f},{target_z:.1f}) yaw={target_yaw_deg:.1f} state={robot_state}")

                action = "PICK" if robot_state == "WAIT_TO_PICK" else "PLACE"
                th = threading.Thread(
                    target=do_motion_pick_or_place,
                    args=(arm, target_x, target_y, target_z, target_yaw_deg, action),
                    daemon=True,
                )
                th.start()

            except Exception as e:
                print(f"[ERROR] Click pipeline failed: {e}")
                recover_if_error(arm)
                robot_state = "WAIT_TO_PICK"

        # Show windows
        cv2.imshow("YOLO + GGCNN (SAFE)", annotated)
        if DEBUG_MODE:
            cv2.imshow("GGCNN Heatmap", last_debug_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            print("[ACTION] Manual recover")
            recover_if_error(arm)
        if key == ord("g"):
            print("[ACTION] Go HOME")
            go_home(arm)

    cap.release()
    cv2.destroyAllWindows()
    try:
        go_home(arm)
        arm.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    main()