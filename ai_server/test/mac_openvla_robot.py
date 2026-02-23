import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from xarm.wrapper import XArmAPI
import time

# --- CONFIGURATION ---
CAMERA_SERVER_IP = "192.168.1.20"  # Change to your PC's IP
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"

ROBOT_IP = "192.168.1.30"  # Change to your xArm IP

# VLA Action Scaling (OpenVLA outputs very small delta values, we need to multiply them)
# Tune this carefully. Too high = robot jumps violently. Too low = robot barely moves.
SCALE_FACTOR_XYZ = 50.0 
GRIPPER_THRESHOLD = 0.5

# The task you want the robot to perform
ROBOT_TASK = "pick up the red object"

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
        arm.set_gripper_position(850, wait=True) # Start with open gripper
        
        # Start at a good hovering position
        arm.set_position(x=300, y=0, z=250, roll=180, pitch=0, yaw=0, speed=100, wait=True)
        
        print("[SUCCESS] Robot connected!")
        return arm
    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return None

def main():
    print("==================================================")
    print("🚀 MAC M1 - OPENVLA CONTINUOUS CONTROL CENTER")
    print("==================================================")
    
    # 1. Connect Robot
    arm = connect_robot()
    if not arm:
        return

    # 2. Load OpenVLA on Mac M1 (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Loading OpenVLA-7B on {device}... This takes memory.")
    
    model_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)
    print("[SUCCESS] OpenVLA Brain is Ready!")

    # 3. Connect to Camera Stream
    print(f"[INFO] Connecting to camera stream...")
    cap = cv2.VideoCapture(STREAM_URL)

    window_name = "OpenVLA Robot Vision"
    cv2.namedWindow(window_name)

    print(f"\n⚡ SYSTEM ACTIVE! Task: '{ROBOT_TASK}'")
    print("Press 's' to START AI Control, 'q' to QUIT/EMERGENCY STOP")

    ai_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Stream lost. Reconnecting...")
            time.sleep(1)
            cap.open(STREAM_URL)
            continue

        # Show status on screen
        status_text = "AI: ACTIVE (Thinking...)" if ai_active else "AI: PAUSED (Press 's' to start)"
        color = (0, 255, 0) if ai_active else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Task: {ROBOT_TASK}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- OPENVLA INFERENCE & CONTROL LOOP ---
        if ai_active:
            # Prepare image for AI
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            prompt = f"In: What action should the robot take to {ROBOT_TASK}?\nOut:"
            
            inputs = processor(prompt, image_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

            # Predict Action
            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # Action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action_raw = action.flatten()
            
            if len(action_raw) >= 7:
                # 1. Calculate Deltas (Movement steps)
                # Note: Eye-in-Hand camera mapping needs to be matched with robot axes
                dx = action_raw[0] * SCALE_FACTOR_XYZ
                dy = action_raw[1] * SCALE_FACTOR_XYZ
                dz = action_raw[2] * SCALE_FACTOR_XYZ
                gripper_cmd = action_raw[6]
                
                print(f"[AI] Move Delta: X:{dx:.1f}, Y:{dy:.1f}, Z:{dz:.1f} | Gripper: {gripper_cmd:.2f}")

                # 2. Read Current Position
                code, current_pose = arm.get_position(is_radian=False)
                if code == 0:
                    curr_x, curr_y, curr_z = current_pose[0], current_pose[1], current_pose[2]
                    
                    # Target Position
                    # Again, calibrate axes: Usually Camera Y = Robot X, Camera X = Robot Y
                    target_x = curr_x + dy 
                    target_y = curr_y + dx 
                    target_z = curr_z + dz 

                    # Prevent crashing into the table
                    if target_z < 50: target_z = 50 
                    
                    # 3. Send Command (wait=False makes it continuous and fluid)
                    arm.set_position(x=target_x, y=target_y, z=target_z, roll=180, pitch=0, yaw=0, speed=100, wait=False)

                    # 4. Gripper Control
                    if gripper_cmd < GRIPPER_THRESHOLD:
                        arm.set_gripper_position(0, wait=False) # Close
                    else:
                        arm.set_gripper_position(850, wait=False) # Open

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 EMERGENCY STOP!")
            break
        elif key == ord('s'):
            ai_active = not ai_active
            print(f"\nToggle AI Control: {'ON' if ai_active else 'OFF'}")
            if not ai_active:
                arm.clean_warn() # Stop moving immediately

    cap.release()
    cv2.destroyAllWindows()
    if arm: 
        arm.set_state(4) # Stop robot
        arm.disconnect()

if __name__ == "__main__":
    main()