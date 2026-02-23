import cv2
import requests
import numpy as np
import math
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import time

# --- CONFIGURATION ---
CAMERA_SERVER_IP = "192.168.1.20"  # Change to your PC's IP
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_3d_coordinate"

ROBOT_IP = "192.168.1.30"  # Change to your xArm IP

# --- EYE-IN-HAND OFFSETS (in millimeters) ---
CAMERA_OFFSET_X = 0.0  
CAMERA_OFFSET_Y = 50.0 
GRASP_HEIGHT = 150.0  
HOVER_HEIGHT = 300.0  

# System States
robot_state = "WAIT_TO_PICK" 
click_target = None
current_grasp_yaw = 0.0 # Stores the angle calculated by AnyGrasp/PCA

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        if x != 0 and y != 0:
            click_target = (x, y)

# --- ANYGRASP (GEOMETRIC PCA) LOGIC ---
def get_orientation_from_mask(mask_coords):
    """
    Takes YOLO mask coordinates and calculates the grasping angle (Yaw)
    using Principal Component Analysis (PCA).
    """
    sz = len(mask_coords)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(sz):
        data_pts[i, 0] = mask_coords[i, 0]
        data_pts[i, 1] = mask_coords[i, 1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Calculate angle in radians, then convert to degrees
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_deg = math.degrees(angle)
    
    # The center of the mask
    center = (int(mean[0,0]), int(mean[0,1]))
    
    return center, angle_deg, eigenvectors

def connect_robot():
    print(f"[INFO] Connecting to xArm at {ROBOT_IP}...")
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        
        arm.set_gripper_enable(True)
        arm.set_gripper_position(850, wait=True)
        arm.set_position(x=300, y=0, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=0, speed=100, wait=True)
        
        print("[SUCCESS] Robot connected and ready!")
        return arm
    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return None

def calculate_target_pose(arm, cam_x_m, cam_y_m):
    dx_mm = cam_x_m * 1000.0
    dy_mm = cam_y_m * 1000.0

    code, current_pose = arm.get_position(is_radian=False)
    if code != 0:
        return None, None

    curr_x, curr_y, curr_z = current_pose[0], current_pose[1], current_pose[2]

    target_x = curr_x + dy_mm + CAMERA_OFFSET_X
    target_y = curr_y - dx_mm + CAMERA_OFFSET_Y
    
    return target_x, target_y

def execute_pick(arm, target_x, target_y, target_yaw):
    global robot_state
    robot_state = "MOVING"

    try:
        print(f"[ACTION] Picking object at X={target_x:.1f}, Y={target_y:.1f}, Yaw={target_yaw:.1f}")
        # 1. Hover over object WITH THE CORRECT ANGLE
        arm.set_position(x=target_x, y=target_y, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=target_yaw, speed=100, wait=True)
        # 2. Descend
        arm.set_position(z=GRASP_HEIGHT, wait=True)
        # 3. Grip
        arm.set_gripper_position(0, wait=True)
        time.sleep(0.5)
        # 4. Lift
        arm.set_position(z=HOVER_HEIGHT, wait=True)
        
        print("[SUCCESS] Object picked up. Please click where to drop it.")
        robot_state = "WAIT_TO_PLACE"
    except Exception as e:
        print(f"[ERROR] Pick failed: {e}")
        robot_state = "WAIT_TO_PICK"

def execute_place(arm, target_x, target_y):
    global robot_state
    robot_state = "MOVING"

    try:
        print(f"[ACTION] Placing object at X={target_x:.1f}, Y={target_y:.1f}")
        # 1. Move to drop location (keep the yaw at 0 for safe placement)
        arm.set_position(x=target_x, y=target_y, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=0, speed=100, wait=True)
        # 2. Descend
        arm.set_position(z=GRASP_HEIGHT, wait=True)
        # 3. Release
        arm.set_gripper_position(850, wait=True)
        time.sleep(0.5)
        # 4. Lift back up
        arm.set_position(z=HOVER_HEIGHT, wait=True)
        
        print("[SUCCESS] Object placed. Ready for next item.")
        robot_state = "WAIT_TO_PICK"
    except Exception as e:
        print(f"[ERROR] Place failed: {e}")
        robot_state = "WAIT_TO_PLACE"

def main():
    print("==================================================")
    print("🚀 MAC M1 SEGMENTATION & ANYGRASP CONTROL CENTER")
    print("==================================================")
    
    arm = connect_robot()

    print("[INFO] Loading YOLOv8-Seg model...")
    model = YOLO("yolov8n-seg.pt") 

    print(f"[INFO] Connecting to camera stream...")
    cap = cv2.VideoCapture(STREAM_URL)

    window_name = "YOLO + AnyGrasp Control"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    global click_target, robot_state, current_grasp_yaw

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Stream lost. Reconnecting...")
            time.sleep(1)
            cap.open(STREAM_URL)
            continue

        # --- YOLO INFERENCE ---
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        # UI Text
        if robot_state == "WAIT_TO_PICK":
            ui_text = "READY - CLICK AN OBJECT TO PICK"
            color = (0, 255, 0)
        elif robot_state == "WAIT_TO_PLACE":
            ui_text = "HOLDING OBJECT - CLICK WHERE TO DROP"
            color = (255, 165, 0)
        else:
            ui_text = "ROBOT MOVING - PLEASE WAIT..."
            color = (0, 0, 255)
            
        cv2.putText(annotated_frame, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- CLICK HANDLING & ANYGRASP LOGIC ---
        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None 
            
            target_yaw = 0.0 # Default angle

            if robot_state == "WAIT_TO_PICK":
                # Find which YOLO mask was clicked
                clicked_inside_mask = False
                for r in results:
                    if r.masks is not None:
                        for mask_xy in r.masks.xy:
                            # Convert YOLO mask to OpenCV contour format
                            contour = np.array(mask_xy, dtype=np.int32).reshape((-1, 1, 2))
                            # Check if the click is inside this contour
                            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                                clicked_inside_mask = True
                                # Calculate PCA Angle (AnyGrasp logic)
                                center, angle_deg, eigenvectors = get_orientation_from_mask(mask_xy)
                                target_yaw = angle_deg
                                
                                # Draw the orientation line on the screen for visual feedback
                                p1 = (int(center[0] + 0.02 * eigenvectors[0,0] * 1000), 
                                      int(center[1] + 0.02 * eigenvectors[0,1] * 1000))
                                cv2.line(annotated_frame, center, p1, (255, 255, 0), 3)
                                cv2.circle(annotated_frame, center, 5, (0, 0, 255), -1)
                                
                                print(f"[INFO] Object Angle (Yaw) Calculated: {target_yaw:.1f} degrees")
                                break
                    if clicked_inside_mask: break
                
                if not clicked_inside_mask:
                    print("[WARNING] You clicked outside an object. Using default 0 degree angle.")

            print(f"[INFO] Querying server for depth at ({cx}, {cy})...")
            try:
                resp = requests.post(API_URL, json={"x": cx, "y": cy}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    target_x, target_y = calculate_target_pose(arm, data["x_m"], data["y_m"])
                    
                    if target_x is not None:
                        if robot_state == "WAIT_TO_PICK":
                            execute_pick(arm, target_x, target_y, target_yaw)
                        elif robot_state == "WAIT_TO_PLACE":
                            # Use 0 angle for placing
                            execute_place(arm, target_x, target_y)
                else:
                    print(f"[WARNING] API Rejected Query: {resp.json().get('error', 'Unknown')}")
            except Exception as e:
                print(f"[ERROR] API Connection Failed: {e}")

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arm: 
        arm.disconnect()

if __name__ == "__main__":
    main()