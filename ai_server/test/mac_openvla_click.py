import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from ultralytics import YOLO
from xarm.wrapper import XArmAPI
import time

# --- CONFIGURATION ---


#CAMERA_SERVER_IP = "10.2.172.118"   
#ROBOT_IP = "10.2.172.20"
CAMERA_SERVER_IP = "192.168.1.20"
ROBOT_IP = "192.168.1.30"
STREAM_URL = f"http://{CAMERA_SERVER_IP}:8000/video_feed"

# OpenVLA Ayarları
SCALE_FACTOR_XYZ = 150.0  # OpenVLA'nın küçük hareketlerini milimetreye çevirme katsayısı
GRIPPER_THRESHOLD = -0.3
SAFE_Z_LIMIT = 50.0      # Masaya çarpmamak için minimum Z yüksekliği (mm)

# Durumlar
robot_state = "SELECTING"  # SELECTING, TRACKING_AND_VLA
click_target = None
tracker = None
tracked_bbox = None
object_name = "object"

def mouse_callback(event, x, y, flags, param):
    global click_target, robot_state
    if event == cv2.EVENT_LBUTTONDOWN and robot_state == "SELECTING":
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
        
        arm.set_gripper_enable(True)
        arm.set_gripper_position(850, wait=True)
        arm.set_position(x=300, y=0, z=400, roll=180, pitch=0, yaw=0, speed=100, wait=False)
        print("[SUCCESS] Robot connected!")
        return arm
    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return None

def main():
    print("==================================================")
    print("🚀 MAC M1 - CLICK TO OPENVLA CONTROL")
    print("==================================================")
    
    global robot_state, click_target, tracker, tracked_bbox, object_name

    arm = connect_robot()

    print("[INFO] Loading YOLO-Seg model...")
    yolo_model = YOLO("models/yolo26m-seg.pt")

    # 2. OpenVLA Yükle (Sürüş için)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Loading OpenVLA-7B on {device}... Please wait.")
    vla_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(vla_id, trust_remote_code=True)
    vla_model = AutoModelForVision2Seq.from_pretrained(
        vla_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    print("[SUCCESS] OpenVLA Brain is Ready!")

    # 3. Kamera Yayını
    print(f"[INFO] Connecting to camera stream...")
    cap = cv2.VideoCapture(STREAM_URL)

    window_name = "Click & OpenVLA Control"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Stream lost. Reconnecting...")
            time.sleep(1)
            cap.open(STREAM_URL)
            continue

        display_frame = frame.copy()

        # ====================================================
        # STATE 1: SEÇİM (SELECTING) - YOLO devrede
        # ====================================================
        if robot_state == "SELECTING":
            results = yolo_model(frame, verbose=False)
            display_frame = results[0].plot(boxes=False) # Sadece maskeleri çiz
            
            cv2.putText(display_frame, "SELECTING: CLICK AN OBJECT", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if click_target:
                cx, cy = click_target
                click_target = None
                
                # Tıklanan objeyi bul
                found = False
                for r in results:
                    if r.masks is not None and r.boxes is not None:
                        for i, mask_xy in enumerate(r.masks.xy):
                            contour = np.array(mask_xy, dtype=np.int32).reshape((-1, 1, 2))
                            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                                # Tıklanan objenin Bounding Box'ını al
                                box = r.boxes.xyxy[i]
                                x1, y1, x2, y2 = map(int, box)
                                tracked_bbox = (x1, y1, x2 - x1, y2 - y1) # (x, y, w, h)
                                
                                # Objenin adını al (Dinamik Prompt için)
                                cls_id = int(r.boxes.cls[i])
                                object_name = yolo_model.names[cls_id]
                                
                                # Takipçiyi (Tracker) başlat
                                
                                tracker = cv2.TrackerMIL.create()
                                tracker.init(frame, tracked_bbox)
                                
                                print(f"[INFO] Selected: {object_name}. Switching to OpenVLA Control.")
                                robot_state = "TRACKING_AND_VLA"
                                found = True
                                break
                    if found: break

        # ====================================================
        # STATE 2: OPENVLA SÜRÜŞ (TRACKING_AND_VLA)
        # ====================================================
        elif robot_state == "TRACKING_AND_VLA":
            # 1. Objeyi Takip Et
            success, bbox = tracker.update(frame)
            
            if success:
                x, y, w, h = map(int, bbox)
                # Objenin etrafına KALIN KIRMIZI KUTU çiz (OpenVLA'nın görmesi için)
                # Orijinal 'frame' üzerine çiziyoruz ki VLA modeli bunu görsün
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
                
                # Kullanıcı ekranı için de çiz
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 6)
                cv2.putText(display_frame, f"VLA DRIVING TO: {object_name.upper()}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 2. OpenVLA'yı Çalıştır
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Kırmızı kutulu resim!
                image_pil = Image.fromarray(image_rgb)
                
                # Prompt: Kırmızı kutunun içindeki [obje_adı]'nı al
                prompt = f"In: What action should the robot take to pick up the {object_name} inside the red bounding box?\nOut:"
                
                inputs = processor(prompt, image_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

                with torch.no_grad():
                    action = vla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                
                    action_raw = action.flatten()
                    
                    # Eğer numpy ise direkt kullan
                    ar = np.array(action_raw).flatten()
                    
                    print("[VLA] len:", ar.shape[0])
                    print("[VLA] vals:", np.round(ar, 3).tolist())
                    
                    if ar.shape[0] > 6:
                        print("[VLA] gripper(raw[6]):", float(ar[6]))
                
                # 3. Robotu Hareket Ettir
                if len(action_raw) >= 7 and arm is not None:
                    dx = action_raw[0] * SCALE_FACTOR_XYZ
                    dy = action_raw[1] * SCALE_FACTOR_XYZ
                    dz = action_raw[2] * SCALE_FACTOR_XYZ
                    gripper_cmd = action_raw[6]

                    code, current_pose = arm.get_position(is_radian=False)
                    if code == 0:
                        curr_x, curr_y, curr_z = current_pose[0], current_pose[1], current_pose[2]
                        
                        # Eksen haritalaması (Kamera robota nasıl takılıysa buna göre +/- yap)
                        target_x = curr_x + dy 
                        target_y = curr_y + dx 
                        target_z = max(curr_z + dz, SAFE_Z_LIMIT) 
                        
                        # Sürekli akıcı hareket (wait=False)
                        arm.set_position(x=target_x, y=target_y, z=target_z, roll=180, pitch=0, yaw=0, speed=100, wait=False)

                        # Eğer model gripper'ı kapatmaya karar verirse
                        if gripper_cmd < GRIPPER_THRESHOLD:
                            print("[INFO] VLA Decided to GRASP! Stopping movement.")
                            arm.set_gripper_position(0, wait=True)
                            time.sleep(1)
                            arm.set_position(z=400, wait=True) # Havaya kaldır
                            robot_state = "SELECTING" # Seçime geri dön
                            arm.set_gripper_position(850, wait=True) # Bırak
            else:
                print("[WARNING] Tracker lost the object. Returning to selection mode.")
                robot_state = "SELECTING"
                if arm: arm.clean_warn() # Hareketi durdur

        # Ekranı göster
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n🛑 ÇIKIŞ YAPILIYOR!")
            break
        elif key == ord('r'):
            # Acil iptal tuşu (R) - Seçime geri döner
            print("\n[INFO] Resetting to Selection Mode.")
            robot_state = "SELECTING"
            if arm: arm.clean_warn()

    cap.release()
    cv2.destroyAllWindows()
    if arm: 
        arm.set_state(4) 
        arm.disconnect()

if __name__ == "__main__":
    main()