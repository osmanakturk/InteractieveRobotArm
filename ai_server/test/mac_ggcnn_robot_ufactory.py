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
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max
from skimage.draw import disk
from ultralytics import YOLO
from xarm.wrapper import XArmAPI

# PyTorch uyarılarını temizle
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch.serialization")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR) 

import models.ggcnn2
from models.ggcnn2 import GGCNN2

# ==============================================================================
# --- ANA AYARLAR (CONFIGURATION) ---
# ==============================================================================
DEBUG_MODE = True                    # GGCNN Isı Haritasını (Heatmap) pencerede gösterir.
USE_UFACTORY_LIMITS = False          # True: UFACTORY'nin masa sınırlarını uygular. False: Sınırları görmezden gelir.
YOLO_MODEL_NAME = "yolo26n-seg.pt"   # Kullanılacak YOLO modeli (Bunu ileride değiştirebilirsin)

CAMERA_SERVER_IP = "192.168.1.20"   
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_grasp_data"
ROBOT_IP = "192.168.1.30"

# --- KİNEMATİK VE OFSETLER ---
CAMERA_OFFSET_X = 0.0  
CAMERA_OFFSET_Y = 50.0 
GRIPPER_Z_MM = 150.0                 # Gripper'ın fiziksel uzunluğu (Dinamik Z hesabı için)

# UFACTORY Orijinal Değerleri
OPEN_LOOP_HEIGHT = 340.0 
GRASPING_RANGE = [180, 600, -300, 300] 
GRASPING_MIN_Z = 175.0               # Masaya çarpmamak için minimum yükseklik
DETECT_XYZ = [300.0, 0.0, 400.0]     # HOME Pozisyonu
RELEASE_XYZ = [400.0, 300.0, 270.0]  # Bırakma Pozisyonu

EULER_EEF_TO_COLOR_OPT = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]
EULER_COLOR_TO_DEPTH_OPT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# ==============================================================================

robot_state = "WAIT_TO_PICK" 
click_target = None
ggcnn_model = None
device = None
prev_mp = np.array([150, 150], dtype=np.int64) 

# --- KİNEMATİK MATEMATİK FONKSİYONLARI ---
def get_sign(x): return 1 if x >= 0 else -1

def rpy_to_rot(rpy_angles):
    Rx = np.asmatrix([[1, 0, 0],[ 0, math.cos(rpy_angles[0]), -math.sin(rpy_angles[0])],[ 0, math.sin(rpy_angles[0]), math.cos(rpy_angles[0])]])
    Ry = np.asmatrix([[math.cos(rpy_angles[1]),0, math.sin(rpy_angles[1])],[ 0, 1, 0],[ -math.sin(rpy_angles[1]), 0, math.cos(rpy_angles[1])]])
    Rz = np.asmatrix([[math.cos(rpy_angles[2]), -math.sin(rpy_angles[2]), 0],[ math.sin(rpy_angles[2]), math.cos(rpy_angles[2]), 0],[ 0, 0, 1]])
    return Rz*Ry*Rx

def rot_to_rpy(RotM):
    pitch = math.atan2(-RotM[2,0], math.sqrt(RotM[2,1]**2 + RotM[2,2]**2))
    sign = get_sign(math.cos(pitch))
    roll = math.atan2(RotM[2,1]*sign, RotM[2,2]*sign)
    yaw = math.atan2(RotM[1,0]*sign, RotM[0,0]*sign)
    return [roll, pitch, yaw]

def euler2mat(euler_vect6d):
    rpy = euler_vect6d[3:]
    rot_3x3 = rpy_to_rot(rpy)
    xyz = np.array([[euler_vect6d[0]],[euler_vect6d[1]],[euler_vect6d[2]]])
    padding_row = np.array([[0,0,0,1]])
    tmp_mat = np.concatenate((rot_3x3,xyz), axis=1)
    return np.concatenate((tmp_mat, padding_row), axis=0)

def mat2euler(mat4x4):
    rot = mat4x4[0:3, 0:3]
    xyz = mat4x4[:3, 3]
    rpy = rot_to_rpy(rot)
    return [xyz[0].item(), xyz[1].item(), xyz[2].item(), rpy[0], rpy[1], rpy[2]]

def convert_pose(pose6d, mat_of_source_in_tar_frame):
    mat_in_source = euler2mat(pose6d)
    mat_in_tar = mat_of_source_in_tar_frame * mat_in_source
    return mat2euler(mat_in_tar)

# --- YAPAY ZEKA VE GÖRÜNTÜ İŞLEME ---
def process_depth_image(depth, out_size=300):
    depth_crop = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)
    depth_crop[depth_nan_mask == 1] = 0

    depth_scale = np.abs(depth_crop).max()
    if depth_scale == 0: depth_scale = 1
    depth_crop = depth_crop.astype(np.float32) / depth_scale

    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
    depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_crop, depth_nan_mask

def load_ggcnn_model():
    global ggcnn_model, device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Loading GGCNN2 Model on ({device})...")
    
    weight_path = os.path.join(SCRIPT_DIR, 'ggcnn_weights.pt')
    try:
        torch.serialization.add_safe_globals([GGCNN2])
        ggcnn_model = torch.load(weight_path, map_location=device, weights_only=False)
        ggcnn_model.to(device)
        ggcnn_model.eval() 
        print(f"[SUCCESS] GGCNN2 Loaded!")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        exit()

def get_grasp_data(depth_image_bytes, current_z_m):
    global prev_mp
    out_size = 300
    
    nparr = np.frombuffer(depth_image_bytes, np.uint8)
    depth_img = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)
    if depth_img is None: return None, None
    depth_img = depth_img.astype(np.float32) * 0.001 

    depth_crop, depth_nan_mask = process_depth_image(depth_img, out_size)

    depth_center = depth_crop[100:141, 130:171].flatten()
    depth_center.sort()
    depth_center_m = depth_center[:10].mean() if len(depth_center) > 10 else depth_crop.mean()

    depth_crop_norm = np.clip((depth_crop - depth_crop.mean()), -1, 1)
    depthT = torch.from_numpy(depth_crop_norm.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    
    with torch.no_grad():
        pred_out = ggcnn_model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask == 1] = 0
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0
    
    points_out = ndimage.gaussian_filter(points_out, 5.0)
    ang_out = ndimage.gaussian_filter(ang_out, 2.0)
    points_out = np.clip(points_out, 0.0, 1.0 - 1e-3)

    if current_z_m > (OPEN_LOOP_HEIGHT / 1000.0):
        max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
        prev_mp = max_pixel.astype(np.int64)
    else:
        maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
        if maxes.shape[0] == 0: return None, None
        
        max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]
        if np.linalg.norm(max_pixel - prev_mp) > 30:
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            prev_mp = max_pixel.astype(np.int64)
        else:
            prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int64)

    ang = ang_out[prev_mp[0], prev_mp[1]]

    debug_img = None
    if DEBUG_MODE:
        debug_img = cv2.applyColorMap((points_out * 255).astype(np.uint8), cv2.COLORMAP_JET)
        rr, cc = disk(prev_mp, 5, shape=(out_size, out_size, 3))
        debug_img[rr, cc, 0] = 0
        debug_img[rr, cc, 1] = 255
        debug_img[rr, cc, 2] = 0
        
        # Hedef açıyı çiz (Kırmızı çizgi)
        line_length = 25
        dx = int(line_length * math.cos(ang))
        dy = int(line_length * math.sin(ang))
        cv2.line(debug_img, (prev_mp[1] - dx, prev_mp[0] - dy), (prev_mp[1] + dx, prev_mp[0] + dy), (0, 0, 255), 3)

        cv2.putText(debug_img, f"Ang: {math.degrees(ang):.1f} | Dp: {depth_center_m*1000:.1f}mm", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {"ang_rad": ang, "depth_center_m": depth_center_m}, debug_img

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        if x != 0 and y != 0: click_target = (x, y)

def go_home(arm):
    arm.set_position(x=DETECT_XYZ[0], y=DETECT_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=150, wait=True)

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
        print(f"[ERROR] {e}")
        return None

def main():
    global click_target, robot_state
    print("==================================================")
    print("🚀 MAC M1 - ULTIMATE GGCNN (FLEXIBLE CONFIG)")
    print("==================================================")
    
    load_ggcnn_model()
    arm = connect_robot()
    if not arm: return

    print(f"[INFO] Loading YOLO Model: {YOLO_MODEL_NAME} ...")
    model = YOLO(YOLO_MODEL_NAME) 
    cap = cv2.VideoCapture(STREAM_URL)
    
    cv2.namedWindow("YOLO + GGCNN")
    cv2.setMouseCallback("YOLO + GGCNN", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(1); cap.open(STREAM_URL); continue

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        if robot_state == "WAIT_TO_PICK":
            cv2.putText(annotated_frame, "HOME - CLICK TO PICK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif robot_state == "WAIT_TO_PLACE":
            cv2.putText(annotated_frame, "HOME - CLICK TO DROP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None 
            
            try:
                resp = requests.post(API_URL, json={"x": cx, "y": cy}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    dx_mm = data["x_m"] * 1000.0
                    dy_mm = data["y_m"] * 1000.0
                    
                    code, current_pose = arm.get_position(is_radian=True)
                    if code != 0: continue
                    eef_pos_m = [current_pose[0]*0.001, current_pose[1]*0.001, current_pose[2]*0.001, current_pose[3], current_pose[4], current_pose[5]]

                    if robot_state == "WAIT_TO_PICK":
                        robot_state = "MOVING"
                        
                        depth_bytes = base64.b64decode(data["depth_image_b64"])
                        ai_result, debug_img = get_grasp_data(depth_bytes, eef_pos_m[2])
                        
                        if debug_img is not None:
                            cv2.imshow("GGCNN Heatmap", debug_img)
                            cv2.waitKey(1)

                        if ai_result is None:
                            print("[ERROR] AI could not find a valid grasp point.")
                            robot_state = "WAIT_TO_PICK"
                            continue

                        # Kinematik Dönüşüm
                        gp = [dx_mm*0.001, dy_mm*0.001, ai_result["depth_center_m"], 0, 0, -1 * ai_result["ang_rad"]]
                        mat_depthOpt_in_base = euler2mat(eef_pos_m) * euler2mat(EULER_EEF_TO_COLOR_OPT) * euler2mat(EULER_COLOR_TO_DEPTH_OPT)
                        gp_base = convert_pose(gp, mat_depthOpt_in_base)

                        ang_final = gp_base[5] - np.pi/2
                        
                        target_x = current_pose[0] * 1000 + dy_mm + CAMERA_OFFSET_X
                        target_y = current_pose[1] * 1000 - dx_mm + CAMERA_OFFSET_Y
                        
                        # Dinamik Z Hesaplama
                        calculated_z = current_pose[2] * 1000 - (ai_result["depth_center_m"] * 1000.0) + GRIPPER_Z_MM

                        # --- SINIR KONTROLLERİ (TOGGLE) ---
                        if USE_UFACTORY_LIMITS:
                            target_z = max(GRASPING_MIN_Z, calculated_z) # Min Z limiti uygula
                            
                            # Menzil limiti kontrolü
                            if target_x < GRASPING_RANGE[0] or target_x > GRASPING_RANGE[1] or target_y < GRASPING_RANGE[2] or target_y > GRASPING_RANGE[3]:
                                print(f"[WARN] Hedef menzil dışında (Sınırlar Aktif): X={target_x:.1f}, Y={target_y:.1f}")
                                robot_state = "WAIT_TO_PICK"
                                continue
                        else:
                            target_z = calculated_z # Sınırlar kapalı, robot tam olarak hesaplanan yere inmeye çalışır
                        # ----------------------------------

                        print(f"[ACTION] PICK: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}, Ang={math.degrees(ang_final):.1f}")
                        
                        arm.set_position(x=target_x, y=target_y, z=DETECT_XYZ[2], roll=180, pitch=0, yaw=math.degrees(ang_final + np.pi), speed=150, wait=True)
                        arm.set_position(z=target_z, speed=100, wait=True)
                        
                        arm.set_gripper_position(0, wait=True)
                        time.sleep(0.5)
                        
                        arm.set_position(z=DETECT_XYZ[2], wait=True)
                        go_home(arm)
                        robot_state = "WAIT_TO_PLACE"
                        
                    elif robot_state == "WAIT_TO_PLACE":
                        robot_state = "MOVING"
                        print(f"[ACTION] PLACE: X={RELEASE_XYZ[0]:.1f}, Y={RELEASE_XYZ[1]:.1f}, Z={RELEASE_XYZ[2]:.1f}")
                        
                        arm.set_position(x=RELEASE_XYZ[0], y=RELEASE_XYZ[1], z=DETECT_XYZ[2], roll=180, pitch=0, yaw=0, speed=150, wait=True)
                        arm.set_position(z=RELEASE_XYZ[2], speed=100, wait=True)
                        
                        arm.set_gripper_position(850, wait=True)
                        time.sleep(0.5)
                        
                        arm.set_position(z=DETECT_XYZ[2], wait=True)
                        go_home(arm)
                        robot_state = "WAIT_TO_PICK"
                            
            except Exception as e:
                print(f"[ERROR] {e}")
                robot_state = "WAIT_TO_PICK" 

        cv2.imshow("YOLO + GGCNN", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    if arm: go_home(arm); arm.disconnect()

if __name__ == "__main__": main()