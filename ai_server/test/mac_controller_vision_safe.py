import os
import time
import math
import base64
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import warnings
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from skimage.draw import disk
from ultralytics import YOLO
from xarm.wrapper import XArmAPI

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")

# =========================
# NETWORK
# =========================
#CAMERA_SERVER_IP = "10.2.172.118"
#ROBOT_IP = "10.2.172.20"

CAMERA_SERVER_IP = "192.168.1.20"
ROBOT_IP = "192.168.1.30"

#STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
#API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_grasp_pack"


# =========================
# MODELS
# =========================
YOLO_MODEL_NAME = "models/yolo26x-seg.pt"
GGCNN_WEIGHT_PATH = "ggcnn_weights.pt"
DEBUG_MODE = True

# =========================
# SAFETY / MODES
# =========================
DRY_RUN_XY_ONLY = False  # <-- FIRST: True (only XY move at VISION_Z). After XY is correct, set False.
RELOAD_H_ON_KEY = True  # enable 'l' to reload homography during runtime

# =========================
# YOUR MEASUREMENTS
# =========================
TABLE_Z_MM = 183.195312

# Safety Z
CLEARANCE_MM = 20.0
MIN_Z = TABLE_Z_MM + CLEARANCE_MM

# Vision Pose (HOME XY + ORI, but higher Z)
VISION_X = 313.177277
VISION_Y = 1.3344
VISION_Z = 400.0
VISION_ROLL  = -179.829107
VISION_PITCH = 0.058213
VISION_YAW   = -1.014021

# Approach safety
MAX_DROP_MM = 220.0

# Pick/Place Z (controlled test values)
# Start safer than 8mm; we will do step-down attempts if needed.
PICK_CLEARANCE_MM = 5.0
PLACE_CLEARANCE_MM = 25.0
PICK_Z_BASE = TABLE_Z_MM + PICK_CLEARANCE_MM
PLACE_Z = TABLE_Z_MM + PLACE_CLEARANCE_MM

# Pick "step-down" attempts (only when DRY_RUN_XY_ONLY=False)
PICK_ATTEMPTS = 3
PICK_STEP_DOWN_MM = 2.0  # each retry: go 2mm lower (clamped by MIN_Z)

# Speeds
SPEED_MOVE = 120
SPEED_DESCEND = 28   # slower = safer
SPEED_ASCEND = 90

# Gripper
GRIPPER_OPEN = 850
GRIPPER_CLOSED = 0

# Homography file (pixel->base XY)
H_FILE = "H_img2base.npy"

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
# UTILS
# =========================
def recover_if_error(arm: XArmAPI) -> None:
    try:
        _, ew = arm.get_err_warn_code()
        if isinstance(ew, list) and len(ew) > 0 and ew[0] != 0:
            print(f"[RECOVER] err={ew[0]} warn={ew[1] if len(ew)>1 else 0}")
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_state(0)
            time.sleep(0.2)
    except Exception:
        pass

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

def go_vision_pose(arm: XArmAPI):
    recover_if_error(arm)
    arm.set_position(
        x=VISION_X, y=VISION_Y, z=VISION_Z,
        roll=VISION_ROLL, pitch=VISION_PITCH, yaw=VISION_YAW,
        speed=SPEED_MOVE, wait=True
    )

def safe_z(z: float) -> float:
    return max(float(MIN_Z), float(z))

def load_homography() -> Optional[np.ndarray]:
    if os.path.exists(H_FILE):
        try:
            H = np.load(H_FILE)
            if H.shape == (3, 3):
                print(f"[INFO] Homography loaded: {H_FILE}")
                return H
        except Exception as e:
            print("[ERROR] Failed loading homography:", e)
    print("[WARN] Homography not found/invalid. XY may be wrong. Run calib_homography.py")
    return None

def project_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    q = q / q[2]
    return float(q[0]), float(q[1])

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

    depth_inp = cv2.inpaint(depth_norm, invalid, 1, cv2.INPAINT_NS) * scale
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
        cv2.putText(debug_img, f"Angle: {angle_deg:.1f} deg", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return {
        "best_u_300": int(prev_mp[1]),
        "best_v_300": int(prev_mp[0]),
        "angle_deg": angle_deg,
        "debug_img": debug_img
    }

# =========================
# STATE
# =========================
robot_state = "WAIT_TO_PICK"
click_target: Optional[Tuple[int, int]] = None
last_click_pixel: Optional[Tuple[int, int]] = None

def mouse_callback(event, x, y, flags, param):
    global click_target, last_click_pixel, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        click_target = (x, y)
        last_click_pixel = (x, y)

def run_action_thread(arm, x, y, z, action):
    """
    Moves to XY at VISION_Z, then optionally descends to Z and gripper action.
    """
    global robot_state
    try:
        robot_state = "MOVING"
        recover_if_error(arm)

        z = safe_z(z)

        # Move XY at VISION_Z (stable)
        arm.set_position(
            x=x, y=y, z=VISION_Z,
            roll=VISION_ROLL, pitch=VISION_PITCH, yaw=VISION_YAW,
            speed=SPEED_MOVE, wait=True
        )

        if DRY_RUN_XY_ONLY:
            print("[DRY_RUN] XY only (no descend).")
            robot_state = "WAIT_TO_PLACE" if action == "PICK" else "WAIT_TO_PICK"
            go_vision_pose(arm)
            return

        drop = VISION_Z - z
        if drop > MAX_DROP_MM:
            print(f"[SAFETY] Drop too large ({drop:.1f}mm) -> CANCEL")
            robot_state = "WAIT_TO_PICK" if action == "PICK" else "WAIT_TO_PLACE"
            return

        # Descend slowly
        arm.set_position(z=z, speed=SPEED_DESCEND, wait=True)

        if action == "PICK":
            arm.set_gripper_position(GRIPPER_CLOSED, wait=True)
            time.sleep(0.25)
            arm.set_position(z=VISION_Z, speed=SPEED_ASCEND, wait=True)
            robot_state = "WAIT_TO_PLACE"
        else:
            arm.set_gripper_position(GRIPPER_OPEN, wait=True)
            time.sleep(0.25)
            arm.set_position(z=VISION_Z, speed=SPEED_ASCEND, wait=True)
            robot_state = "WAIT_TO_PICK"

        go_vision_pose(arm)

    except Exception as e:
        print("[ERROR] Motion failed:", e)
        recover_if_error(arm)
        robot_state = "WAIT_TO_PICK"

def main():
    global click_target, robot_state, last_debug_img

    print("==================================================")
    print("✅ mac_controller_vision_safe (UPDATED)")
    print("==================================================")
    print(f"[MODE] DRY_RUN_XY_ONLY={DRY_RUN_XY_ONLY}")
    print(f"[VISION] X={VISION_X:.1f} Y={VISION_Y:.1f} Z={VISION_Z:.1f}")
    print(f"[TABLE]  Z={TABLE_Z_MM:.1f}  MIN_Z={MIN_Z:.1f}  PICK_Z_BASE={PICK_Z_BASE:.1f}  PLACE_Z={PLACE_Z:.1f}")
    print("HOTKEYS: q=quit | v=vision pose | e=recover | l=reload H | t=test move to last click (XY) | o=open | c=close")

    ggcnn_model, device = load_ggcnn_model()
    yolo = YOLO(YOLO_MODEL_NAME)
    H = load_homography()

    arm = connect_robot()
    go_vision_pose(arm)

    cap = cv2.VideoCapture(STREAM_URL)
    cv2.namedWindow("YOLO + GGCNN (VISION SAFE)")
    cv2.setMouseCallback("YOLO + GGCNN (VISION SAFE)", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            cap.open(STREAM_URL)
            continue

        results = yolo(frame, verbose=False)
        annotated = results[0].plot()

        if robot_state == "WAIT_TO_PICK":
            cv2.putText(annotated, "CLICK OBJECT TO PICK", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        elif robot_state == "WAIT_TO_PLACE":
            cv2.putText(annotated, "CLICK LOCATION TO PLACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        else:
            cv2.putText(annotated, "ROBOT MOVING...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(annotated, f"DRY_RUN_XY_ONLY={DRY_RUN_XY_ONLY}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None

            # Always stabilize view
            go_vision_pose(arm)
            time.sleep(0.15)

            # Validate click in bbox (only for pick)
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
                resp = requests.post(API_URL, json={"x": int(cx), "y": int(cy), "crop_size": 300}, timeout=3)
                if resp.status_code != 200:
                    print("[ERROR] Server:", resp.status_code, resp.text)
                    continue
                pack = resp.json()
                if not pack.get("valid"):
                    print("[ERROR] Pack invalid:", pack)
                    continue

                depth_scale = float(pack["depth_scale"])
                origin = pack["crop_origin"]

                depth_png = base64.b64decode(pack["depth_crop_b64"])
                depth_u16 = cv2.imdecode(np.frombuffer(depth_png, np.uint8), cv2.IMREAD_ANYDEPTH)
                if depth_u16 is None:
                    print("[ERROR] depth decode failed")
                    continue

                depth_crop_m = depth_u16.astype(np.float32) * depth_scale

                g = ggcnn_infer(depth_crop_m, ggcnn_model, device, out_size=300)
                if g is None:
                    print("[ERROR] GGCNN: no grasp found")
                    continue
                if g["debug_img"] is not None:
                    last_debug_img = g["debug_img"]

                crop_h, crop_w = depth_u16.shape[:2]
                u_in_crop = int(g["best_u_300"] / 300.0 * crop_w)
                v_in_crop = int(g["best_v_300"] / 300.0 * crop_h)

                u_full = int(origin["x"] + u_in_crop)
                v_full = int(origin["y"] + v_in_crop)

                # XY via homography
                if H is not None:
                    tx, ty = project_xy(H, u_full, v_full)
                else:
                    code, pose = arm.get_position(is_radian=False)
                    if code != 0:
                        continue
                    tx, ty = float(pose[0]), float(pose[1])
                    print("[WARN] No homography: holding current XY.")

                action = "PICK" if robot_state == "WAIT_TO_PICK" else "PLACE"

                if action == "PLACE":
                    tz = PLACE_Z
                    print(f"[TARGET] pixel=({u_full},{v_full}) -> XY=({tx:.1f},{ty:.1f}) Z={tz:.1f} action=PLACE")
                    th = threading.Thread(target=run_action_thread, args=(arm, tx, ty, tz, "PLACE"), daemon=True)
                    th.start()
                else:
                    # PICK with step-down attempts
                    # We do not “sense” grasp success yet, but this helps with height mismatch.
                    for attempt in range(PICK_ATTEMPTS):
                        tz = PICK_Z_BASE - attempt * PICK_STEP_DOWN_MM
                        tz = safe_z(tz)
                        print(f"[TARGET] pixel=({u_full},{v_full}) -> XY=({tx:.1f},{ty:.1f}) Z={tz:.1f} action=PICK attempt={attempt+1}/{PICK_ATTEMPTS}")
                        th = threading.Thread(target=run_action_thread, args=(arm, tx, ty, tz, "PICK"), daemon=True)
                        th.start()
                        break  # we launch one attempt; if you want sequential attempts, we can chain it later.

            except Exception as e:
                print("[ERROR] click pipeline failed:", e)
                recover_if_error(arm)
                robot_state = "WAIT_TO_PICK"

        cv2.imshow("YOLO + GGCNN (VISION SAFE)", annotated)
        if DEBUG_MODE:
            cv2.imshow("GGCNN Heatmap", last_debug_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("v"):
            print("[ACTION] go vision pose")
            go_vision_pose(arm)

        if key == ord("e"):
            print("[ACTION] recover")
            recover_if_error(arm)

        if key == ord("l") and RELOAD_H_ON_KEY:
            print("[ACTION] reload homography")
            H = load_homography()

        if key == ord("t"):
            # test move using last click pixel (no ggcnn)
            if last_click_pixel is None:
                print("[WARN] No click yet.")
            elif H is None:
                print("[WARN] No homography loaded.")
            else:
                u, v = last_click_pixel
                tx, ty = project_xy(H, u, v)
                print(f"[TEST] pixel=({u},{v}) -> XY=({tx:.1f},{ty:.1f}) at VISION_Z")
                th = threading.Thread(target=run_action_thread, args=(arm, tx, ty, VISION_Z, "PLACE"), daemon=True)
                th.start()

        if key == ord("o"):
            print("[ACTION] open gripper")
            arm.set_gripper_position(GRIPPER_OPEN, wait=True)

        if key == ord("c"):
            print("[ACTION] close gripper")
            arm.set_gripper_position(GRIPPER_CLOSED, wait=True)

    cap.release()
    cv2.destroyAllWindows()
    try:
        go_vision_pose(arm)
        arm.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    main()