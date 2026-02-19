import cv2
import numpy as np
import math
import time
from xarm.wrapper import XArmAPI

# --- AYARLAR ---
ROBOT_IP = "192.168.1.30"   # Loglarda gördüğüm senin robot IP'n bu
CAMERA_INDEX = 0            # RealSense indexi (Görüntü yoksa 0 yap)

def get_orientation(contour, img):
    sz = len(contour)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contour[i,0,0]
        data_pts[i,1] = contour[i,0,1]
    
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_deg = math.degrees(angle)
    return cntr, angle_deg, eigenvectors

def main():
    print("🚀 AnyGrasp (Geometrik Mod - V2) Başlatılıyor...")
    
    print(f"🤖 Robot ({ROBOT_IP}) bağlanıyor...")
    try:
        arm = XArmAPI(ROBOT_IP)
        
        # --- HATA TEMİZLEME (Code 3 Çözümü) ---
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        time.sleep(1)
        
        print("📍 Robot başlangıç pozisyonuna gidiyor...")
        arm.set_position(x=300, y=0, z=250, roll=180, pitch=0, yaw=0, speed=50, wait=True)
        arm.set_gripper_enable(True)
        arm.set_gripper_position(850, wait=True) 
        print("✅ Robot Hazır ve Hatalar Temizlendi!")
    except Exception as e:
        print(f"❌ Robot Hatası: {e}")
        return

    print("👁️ Kamera başlatılıyor...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ Kamera 1 yok, 0 deneniyor...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ HATA: Kamera bulunamadı!")
            return

    print("\n⚡ SİSTEM AKTİF! (Çıkış: 'q', Tutuş: 'g')")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_center = None
            target_angle = 0
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(c) > 2000:
                    target_center, target_angle, eigenvectors = get_orientation(c, frame)
                    
                    # Çizimler
                    cv2.circle(frame, target_center, 8, (0, 0, 255), -1)
                    
                    p1 = (target_center[0] + 0.02 * eigenvectors[0,0] * 1000, target_center[1] + 0.02 * eigenvectors[0,1] * 1000)
                    cv2.line(frame, target_center, (int(p1[0]), int(p1[1])), (0, 255, 0), 3)
                    
                    label = f"Aci: {target_angle:.1f}"
                    cv2.putText(frame, label, (target_center[0] + 20, target_center[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # --- HATA DÜZELTME BURADA ---
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box) # <--- np.int0 yerine np.int32 kullanıldı!
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            cv2.imshow('AnyGrasp Fix', frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('g') and target_center is not None:
                print(f"\n🦾 HAREKET: Merkez={target_center}, Açı={target_angle:.1f}")
                
                img_w, img_h = 640, 480
                center_x, center_y = target_center
                
                rel_x = (center_x - img_w/2) 
                rel_y = (center_y - img_h/2)
                
                code, current_pos = arm.get_position(is_radian=False)
                curr_x, curr_y, curr_z = current_pos[0], current_pos[1], current_pos[2]
                
                # Koordinat Mapping (Hassasiyet ayarı: 0.5)
                target_robot_y = curr_y - (rel_x * 0.5) 
                target_robot_x = curr_x - (rel_y * 0.5)
                
                print(f"📍 Hedef: X={target_robot_x:.1f}, Y={target_robot_y:.1f}")
                
                # 1. Git
                arm.set_position(x=target_robot_x, y=target_robot_y, z=curr_z, 
                                 roll=180, pitch=0, yaw=target_angle, speed=100, wait=True)
                # 2. İn
                arm.set_position(z=160, wait=True) # Güvenli yükseklik
                # 3. Tut
                arm.set_gripper_position(0, wait=True)
                time.sleep(0.5)
                # 4. Kaldır
                arm.set_position(z=300, wait=True)
                
                print("✅ Tamamlandı. Bırakılıyor...")
                time.sleep(1)
                arm.set_gripper_position(850, wait=True)

    except KeyboardInterrupt:
        print("🛑 Durduruluyor...")
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()