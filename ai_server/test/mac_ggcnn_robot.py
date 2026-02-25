import os
import sys
import cv2
import requests
import numpy as np
import base64
import torch
import math
import time
import warnings
import threading  
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from skimage.draw import disk
from ultralytics import YOLO
from xarm.wrapper import XArmAPI

# Suppress PyTorch version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR) 

import models.ggcnn2
from models.ggcnn2 import GGCNN2

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
DEBUG_MODE = True                    # Shows the GGCNN Heatmap window
YOLO_MODEL_NAME = "yolo26s-seg.pt"   # Selected YOLO model

CAMERA_SERVER_IP = "192.168.1.20"   
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_grasp_data"
ROBOT_IP = "192.168.1.30"

# --- EYE-IN-HAND OFFSETS & KINEMATICS ---
CAMERA_OFFSET_X = 0.0  
CAMERA_OFFSET_Y = 50.0 
GRIPPER_Z_MM = 150.0                 # Physical length of the gripper for dynamic Z
GRASPING_MIN_Z = 175.0               # Minimum safe Z height to avoid table collision

HOVER_HEIGHT = 400.0  
HOME_X = 300.0
HOME_Y = 0.0
HOME_Z = HOVER_HEIGHT
# ==============================================================================

robot_state = "WAIT_TO_PICK" 
click_target = None
ggcnn_model = None
device = None
prev_mp = np.array([150, 150], dtype=np.int64) 

# Initial empty heatmap to prevent window freezing
last_debug_img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(last_debug_img, "WAITING FOR AI...", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def process_depth_image(depth, out_size=300):
    """Fills empty depth pixels (NaNs) using OpenCV Inpainting and resizes the image."""
    depth_crop = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
    depth_crop[depth_nan_mask == 1] = 0

    depth_scale = np.abs(depth_crop).max()
    if depth_scale == 0: depth_scale = 1
    depth_crop = depth_crop.astype(np.float32) / depth_scale

    # Inpaint missing data
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
    depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_crop, depth_nan_mask

def load_ggcnn_model():
    """Loads the pre-trained PyTorch GGCNN2 model."""
    global ggcnn_model, device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Loading GGCNN2 Model on ({device})...")
    
    weight_path = os.path.join(SCRIPT_DIR, 'ggcnn_weights.pt')
    try:
        torch.serialization.add_safe_globals([models.ggcnn2.GGCNN2])
        ggcnn_model = torch.load(weight_path, map_location=device, weights_only=False)
        ggcnn_model.to(device)
        ggcnn_model.eval() 
        print(f"[SUCCESS] GGCNN2 Loaded Successfully!")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        exit()

def get_grasp_data(depth_image_bytes, accurate_depth_mm):
    """Processes the depth map through GGCNN to find the best grasp angle and generates a blended heatmap."""
    global prev_mp
    out_size = 300
    
    # 1. Decode Image
    nparr = np.frombuffer(depth_image_bytes, np.uint8)
    depth_img = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)
    if depth_img is None: return None, None
    depth_img = depth_img.astype(np.float32) * 0.001 

    # 2. Preprocess
    depth_crop, depth_nan_mask = process_depth_image(depth_img, out_size)
    depth_crop_norm = np.clip((depth_crop - depth_crop.mean()), -1, 1)
    depthT = torch.from_numpy(depth_crop_norm.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    
    # 3. AI Inference
    with torch.no_grad():
        pred_out = ggcnn_model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask == 1] = 0
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0
    
    # 4. Post-processing Filters
    points_out = ndimage.gaussian_filter(points_out, 5.0)
    ang_out = ndimage.gaussian_filter(ang_out, 2.0)
    points_out = np.clip(points_out, 0.0, 1.0 - 1e-3)

    # 5. Find Local Maxima (Best Grasp Points)
    maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
    if maxes.shape[0] == 0: return None, None
    
    # Prefer the peak closest to the previous grasp point for stability
    max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]
    if np.linalg.norm(max_pixel - prev_mp) > 30:
        max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
        prev_mp = max_pixel.astype(np.int64)
    else:
        prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int64)

    ang = ang_out[prev_mp[0], prev_mp[1]]
    angle_deg = math.degrees(ang)

    # 6. ENHANCED DEBUG VISUALIZATION (Blended Heatmap)
    debug_img = None
    if DEBUG_MODE:
        # Normalize the raw depth shape to grayscale for background
        depth_vis = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
        # Create the colorful AI Quality Heatmap
        heatmap_bgr = cv2.applyColorMap((points_out * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # BLEND them together (40% physical object, 60% AI Heatmap)
        debug_img = cv2.addWeighted(depth_vis_bgr, 0.4, heatmap_bgr, 0.6, 0)
        
        # Draw Grasp Center (Green Circle)
        rr, cc = disk(prev_mp, 5, shape=(out_size, out_size, 3))
        debug_img[rr, cc, 0] = 0
        debug_img[rr, cc, 1] = 255
        debug_img[rr, cc, 2] = 0
        
        # Draw Grasp Angle (Red Line)
        line_length = 25
        dx = int(line_length * math.cos(ang))
        dy = int(line_length * math.sin(ang))
        cv2.line(debug_img, (prev_mp[1] - dx, prev_mp[0] - dy), (prev_mp[1] + dx, prev_mp[0] + dy), (0, 0, 255), 3)

        # UI Overlay Text
        cv2.putText(debug_img, "AI GRASP QUALITY", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(debug_img, "Red = Best | Blue = Worst", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Bottom Info Box
        cv2.rectangle(debug_img, (0, out_size - 40), (out_size, out_size), (0, 0, 0), -1)
        cv2.putText(debug_img, f"Target Angle : {angle_deg:.1f} deg", (5, out_size - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(debug_img, f"Target Depth : {accurate_depth_mm:.1f} mm", (5, out_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return angle_deg, debug_img

# ==============================================================================
# --- BACKGROUND ROBOT THREAD ---
# ==============================================================================
def perform_robot_action(arm, action_type, target_x, target_y, target_z, target_yaw=0.0):
    global robot_state
    try:
        if action_type == "PICK":
            arm.set_position(x=target_x, y=target_y, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=target_yaw, speed=150, wait=True)
            arm.set_position(z=target_z, speed=100, wait=True)
            
            print("[ACTION] Gripper Closing...")
            arm.set_gripper_position(0, wait=True)
            time.sleep(0.5)
            
            arm.set_position(z=HOVER_HEIGHT, wait=True)
            go_home(arm)
            robot_state = "WAIT_TO_PLACE" 
            
        elif action_type == "PLACE":
            arm.set_position(x=target_x, y=target_y, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=0, speed=150, wait=True)
            arm.set_position(z=GRASPING_MIN_Z, speed=100, wait=True)
            
            print("[ACTION] Gripper Opening...")
            arm.set_gripper_position(850, wait=True)
            time.sleep(0.5)
            
            arm.set_position(z=HOVER_HEIGHT, wait=True)
            go_home(arm)
            robot_state = "WAIT_TO_PICK" 
            
    except Exception as e:
        print(f"[ERROR] Robot Motion Failed: {e}")
        robot_state = "WAIT_TO_PICK" 

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        if x != 0 and y != 0: click_target = (x, y)

def go_home(arm):
    arm.set_position(x=HOME_X, y=HOME_Y, z=HOME_Z, roll=180, pitch=0, yaw=0, speed=150, wait=True)

def connect_robot():
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        arm.set_gripper_enable(True)
        arm.set_gripper_position(850, wait=True)
        go_home(arm)
        return arm
    except Exception as e:
        print(f"[ERROR] Connection Failed: {e}")
        return None

def main():
    global click_target, robot_state, last_debug_img
    print("==================================================")
    print("🚀 MAC M1 - GGCNN (FULL ENGLISH & SMART UI)")
    print("==================================================")
    
    load_ggcnn_model()
    arm = connect_robot()
    if not arm: return

    print(f"[INFO] Loading YOLO Model: {YOLO_MODEL_NAME} ...")
    model = YOLO(YOLO_MODEL_NAME) 
    cap = cv2.VideoCapture(STREAM_URL)
    
    cv2.namedWindow("YOLO + GGCNN")
    cv2.setMouseCallback("YOLO + GGCNN", mouse_callback)

    warning_msg = ""
    warning_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(1); cap.open(STREAM_URL); continue

        # YOLO Inference
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=True) 

        # UI Overlays
        if robot_state == "WAIT_TO_PICK":
            cv2.putText(annotated_frame, "HOME - CLICK ON AN OBJECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif robot_state == "WAIT_TO_PLACE":
            cv2.putText(annotated_frame, "HOME - CLICK TO DROP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif robot_state == "MOVING":
            cv2.putText(annotated_frame, "ROBOT MOVING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show warning message for 2 seconds if missed
        if time.time() - warning_time < 2:
            cv2.putText(annotated_frame, warning_msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None 
            
            # --- 1. YOLO BOUNDING BOX VALIDATION ---
            if robot_state == "WAIT_TO_PICK":
                is_valid_target = False
                target_name = ""
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Check if click coordinates are inside the bounding box
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            is_valid_target = True
                            class_id = int(box.cls[0].item())
                            target_name = model.names[class_id]
                            break
                
                # Cancel action if clicked on empty space
                if not is_valid_target:
                    print(f"[WARN] Action Canceled: No object found at ({cx}, {cy})!")
                    warning_msg = "WARNING: CLICKED ON EMPTY SPACE!"
                    warning_time = time.time()
                    continue 
                
                print(f"\n[INFO] Target Validated: {target_name.upper()} detected!")
            
            # --- 2. REQUEST DATA AND INITIATE GRASP ---
            try:
                resp = requests.post(API_URL, json={"x": cx, "y": cy}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    dx_mm = data["x_m"] * 1000.0
                    dy_mm = data["y_m"] * 1000.0
                    depth_mm = data["z_m"] * 1000.0  
                    
                    code, current_pose = arm.get_position(is_radian=False)
                    if code != 0: continue

                    if robot_state == "WAIT_TO_PICK":
                        robot_state = "MOVING" 
                        
                        depth_bytes = base64.b64decode(data["depth_image_b64"])
                        target_yaw, debug_img = get_grasp_data(depth_bytes, depth_mm)
                        
                        if debug_img is not None:
                            last_debug_img = debug_img

                        if target_yaw is None:
                            print("[ERROR] AI could not find a valid grasp point.")
                            robot_state = "WAIT_TO_PICK"
                            continue

                        target_x = current_pose[0] + dy_mm + CAMERA_OFFSET_X
                        target_y = current_pose[1] - dx_mm + CAMERA_OFFSET_Y
                        
                        # Dynamic Z Landing Formula
                        calculated_z = current_pose[2] - depth_mm + GRIPPER_Z_MM
                        target_z = max(GRASPING_MIN_Z, calculated_z)

                        print(f"[ACTION] PICK: Object={target_name.upper()}, X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}, Yaw={target_yaw:.1f}")

                        t = threading.Thread(target=perform_robot_action, args=(arm, "PICK", target_x, target_y, target_z, target_yaw))
                        t.start()
                        
                    elif robot_state == "WAIT_TO_PLACE":
                        robot_state = "MOVING" 
                        
                        target_x = current_pose[0] + dy_mm + CAMERA_OFFSET_X
                        target_y = current_pose[1] - dx_mm + CAMERA_OFFSET_Y
                        
                        print(f"[ACTION] PLACE: X={target_x:.1f}, Y={target_y:.1f}")
                        
                        t = threading.Thread(target=perform_robot_action, args=(arm, "PLACE", target_x, target_y, 0))
                        t.start()
                            
            except Exception as e:
                print(f"[ERROR] Communication/AI Error: {e}")
                robot_state = "WAIT_TO_PICK" 

        cv2.imshow("YOLO + GGCNN", annotated_frame)
        if DEBUG_MODE:
            cv2.imshow("GGCNN Heatmap", last_debug_img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    if arm: go_home(arm); arm.disconnect()

if __name__ == "__main__": main()