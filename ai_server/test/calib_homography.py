import json
import time
from pathlib import Path

import cv2
import numpy as np
from xarm.wrapper import XArmAPI

# =========================
# CONFIG
# =========================
ROBOT_IP = "10.2.172.20"
STREAM_URL = "http://10.2.172.118:8000/video_feed"

OUT_JSON = Path("homography_points.json")
OUT_H = Path("H_img2base.npy")

# ---- Vision Pose (your chosen stable pose) ----
VISION_X = 313.177277
VISION_Y = 1.3344
VISION_Z = 400.0
VISION_ROLL  = -179.829107
VISION_PITCH = 0.058213
VISION_YAW   = -1.014021

# Robot motion speeds (safe for calibration)
SPEED_MOVE = 120

# =========================
# STATE
# =========================
points = []          # list of {"u":..,"v":..,"x":..,"y":..}
click_uv = None      # (u,v) from mouse click

# =========================
# HELPERS
# =========================
def mouse_cb(event, x, y, flags, param):
    global click_uv
    if event == cv2.EVENT_LBUTTONDOWN:
        click_uv = (int(x), int(y))

def connect_robot():
    arm = XArmAPI(ROBOT_IP)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    return arm

def recover(arm):
    try:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_state(0)
        time.sleep(0.2)
        return True
    except Exception as e:
        print("[ERROR] recover failed:", e)
        return False

def go_vision_pose(arm):
    recover(arm)
    print("[ACTION] Going to VISION_POSE...")
    code = arm.set_position(
        x=VISION_X, y=VISION_Y, z=VISION_Z,
        roll=VISION_ROLL, pitch=VISION_PITCH, yaw=VISION_YAW,
        speed=SPEED_MOVE, wait=True
    )
    if code != 0:
        print("[WARN] set_position returned code:", code)

def get_xy(arm):
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        return None
    return float(pose[0]), float(pose[1])

def print_pose(arm):
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        print("[ERROR] get_position failed")
        return
    x, y, z, r, p, yw = pose
    print(f"[POSE] x={x:.3f} y={y:.3f} z={z:.3f} roll={r:.3f} pitch={p:.3f} yaw={yw:.3f}")

def save_points():
    OUT_JSON.write_text(json.dumps(points, indent=2), encoding="utf-8")
    print(f"[SAVE] {OUT_JSON} ({len(points)} samples)")

def compute_homography():
    if len(points) < 4:
        print("[WARN] Need at least 4 samples.")
        return None

    src = np.array([[p["u"], p["v"]] for p in points], dtype=np.float32)
    dst = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        print("[ERROR] Homography failed.")
        return None

    np.save(str(OUT_H), H)
    print(f"[OK] Saved homography -> {OUT_H}")
    print(H)
    return H

def draw_overlay(frame):
    global click_uv
    y = 25
    dy = 24

    lines = [
        "HOTKEYS:",
        "  v: go VISION_POSE   |  a: add sample   |  u: undo last",
        "  c: compute/save H   |  p: print pose   |  e: recover errors",
        "  q: quit",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"samples: {len(points)}", (10, y + len(lines) * dy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if click_uv is not None:
        cv2.circle(frame, click_uv, 6, (0, 255, 0), 2)
        cv2.putText(frame, f"click: {click_uv}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# =========================
# MAIN
# =========================
def main():
    global click_uv, points

    # Load previous points if exist
    if OUT_JSON.exists():
        try:
            points = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded existing points from {OUT_JSON} ({len(points)})")
        except Exception:
            print("[WARN] Could not parse existing JSON, starting fresh.")
            points = []

    arm = connect_robot()

    cap = cv2.VideoCapture(STREAM_URL)
    cv2.namedWindow("CALIB_H (VISION)")
    cv2.setMouseCallback("CALIB_H (VISION)", mouse_cb)

    print("\n=== HOMOGRAPHY CALIBRATION (VISION POSE) ===")
    print("Goal: map pixel (u,v) -> base (x,y) using a fixed camera view.")
    print("Workflow per point:")
    print("  1) Press 'v' to go to VISION_POSE (stable camera)")
    print("  2) Click the table marker on the image (u,v)")
    print("  3) Jog robot TCP to the SAME marker on table (do not crash)")
    print("  4) Press 'a' to record (u,v) + current robot (x,y)")
    print("After 4+ points, press 'c' to compute/save H.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            cap.open(STREAM_URL)
            continue

        draw_overlay(frame)
        cv2.imshow("CALIB_H (VISION)", frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        if k == ord("v"):
            go_vision_pose(arm)
            continue

        if k == ord("e"):
            print("[ACTION] Recover errors/warnings...")
            recover(arm)
            continue

        if k == ord("p"):
            print_pose(arm)
            continue

        if k == ord("u"):
            if len(points) == 0:
                print("[WARN] No samples to undo.")
                continue
            removed = points.pop()
            save_points()
            print("[UNDO] removed:", removed)
            continue

        if k == ord("a"):
            if click_uv is None:
                print("[WARN] Click a marker first.")
                continue
            xy = get_xy(arm)
            if xy is None:
                print("[ERROR] Could not read robot position. Try 'e' then retry.")
                continue

            u, v = click_uv
            x, y = xy

            sample = {"u": int(u), "v": int(v), "x": float(x), "y": float(y)}
            points.append(sample)
            save_points()
            print("[ADD]", sample)

            click_uv = None
            continue

        if k == ord("c"):
            compute_homography()
            continue

    cap.release()
    cv2.destroyAllWindows()
    try:
        arm.disconnect()
    except Exception:
        pass

    print("[DONE]")

if __name__ == "__main__":
    main()