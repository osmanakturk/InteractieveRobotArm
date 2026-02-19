import cv2
import requests
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import time

# --- CONFIGURATION ---
CAMERA_SERVER_IP = "192.168.1.20"  # Server IP
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"
API_URL = f"http://{CAMERA_SERVER_IP}:8000/api/get_3d_coordinate"

ROBOT_IP = "192.168.1.30"  # xArm IP

# --- EYE-IN-HAND OFFSETS (in millimeters) ---
CAMERA_OFFSET_X = 0.0  
CAMERA_OFFSET_Y = 50.0 
GRASP_HEIGHT = 150.0  # Height to go down to grab/drop (Tune to your table height)
HOVER_HEIGHT = 300.0  # Safe height to travel

# System States
robot_state = "WAIT_TO_PICK" # Can be: "WAIT_TO_PICK", "WAIT_TO_PLACE", "MOVING"
click_target = None

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    # Only accept clicks if we are not currently moving
    if event == cv2.EVENT_LBUTTONDOWN and robot_state != "MOVING":
        if x != 0 and y != 0:
            click_target = (x, y)

def connect_robot():
    print(f"[INFO] Connecting to xArm at {ROBOT_IP}...")
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        
        # Initialize
        arm.set_gripper_enable(True)
        arm.set_gripper_position(850, wait=True) # Open
        arm.set_position(x=300, y=0, z=HOVER_HEIGHT, roll=180, pitch=0, yaw=0, speed=100, wait=True)
        
        print("[SUCCESS] Robot connected and ready!")
        return arm
    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return None

def calculate_target_pose(arm, cam_x_m, cam_y_m):
    """ Converts relative camera meters to absolute robot millimeters """
    dx_mm = cam_x_m * 1000.0
    dy_mm = cam_y_m * 1000.0

    code, current_pose = arm.get_position(is_radian=False)
    if code != 0:
        return None, None, None

    curr_x, curr_y, curr_z = current_pose[0], current_pose[1], current_pose[2]

    # Camera relative mapping (Eye-in-Hand looking down)
    target_x = curr_x + dy_mm + CAMERA_OFFSET_X
    target_y = curr_y - dx_mm + CAMERA_OFFSET_Y
    
    return target_x, target_y, curr_z

def execute_pick(arm, target_x, target_y, curr_z):
    global robot_state
    robot_state = "MOVING"

    try:
        print(f"[ACTION] Picking object at X={target_x:.1f}, Y={target_y:.1f}")
        # 1. Hover over object
        arm.set_position(x=target_x, y=target_y, z=curr_z, roll=180, pitch=0, yaw=0, speed=100, wait=True)
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

def execute_place(arm, target_x, target_y, curr_z):
    global robot_state
    robot_state = "MOVING"

    try:
        print(f"[ACTION] Placing object at X={target_x:.1f}, Y={target_y:.1f}")
        # 1. Move to drop location
        arm.set_position(x=target_x, y=target_y, z=curr_z, roll=180, pitch=0, yaw=0, speed=100, wait=True)
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
    print("🚀 MAC M1 SEGMENTATION & CONTROL CENTER")
    print("==================================================")
    
    arm = connect_robot()

    # Load YOLO Segmentation Model
    print("[INFO] Loading YOLOv8-Seg model...")
    model = YOLO("models/yolo26n-seg.pt") 

    print(f"[INFO] Connecting to camera stream...")
    cap = cv2.VideoCapture(STREAM_URL)

    window_name = "YOLO Segmentation & Control"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    global click_target, robot_state

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Stream lost. Reconnecting...")
            time.sleep(1)
            cap.open(STREAM_URL)
            continue

        # --- YOLO INFERENCE ---
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False) # Only masks, no boxes

        # Show UI Text based on state
        if robot_state == "WAIT_TO_PICK":
            ui_text = "READY - CLICK AN OBJECT TO PICK"
            color = (0, 255, 0) # Green
        elif robot_state == "WAIT_TO_PLACE":
            ui_text = "HOLDING OBJECT - CLICK WHERE TO DROP"
            color = (255, 165, 0) # Orange
        else:
            ui_text = "ROBOT MOVING - PLEASE WAIT..."
            color = (0, 0, 255) # Red
            
        cv2.putText(annotated_frame, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- CLICK HANDLING ---
        if click_target and robot_state != "MOVING":
            cx, cy = click_target
            click_target = None # Reset click immediately to prevent loops
            
            print(f"\n[INFO] Querying server for pixel ({cx}, {cy})...")
            try:
                resp = requests.post(API_URL, json={"x": cx, "y": cy}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Calculate absolute target coords
                    target_x, target_y, curr_z = calculate_target_pose(arm, data["x_m"], data["y_m"])
                    
                    if target_x is not None:
                        # Decide action based on current state
                        if robot_state == "WAIT_TO_PICK":
                            execute_pick(arm, target_x, target_y, curr_z)
                        elif robot_state == "WAIT_TO_PLACE":
                            execute_place(arm, target_x, target_y, curr_z)
                else:
                    error_msg = resp.json().get('error', 'Unknown')
                    print(f"[WARNING] API Rejected Query: {error_msg}")
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