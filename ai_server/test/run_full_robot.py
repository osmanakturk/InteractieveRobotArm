import sys
import os
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# xArm SDK yolunu ekle (Eğer kurulu değilse pip install xarm-python-sdk)
from xarm.wrapper import XArmAPI

# --- AYARLAR ---
ROBOT_IP = "192.168.1.30" # <-- Robotun IP adresini buraya yaz!
SCALE_FACTOR = 80          # Modelin küçük hareketlerini büyütme katsayısı (Hız ayarı)
GRIPPER_THRESHOLD = 0.5    # Kıskaç açma/kapama eşiği

def main():
    print("🚀 SİSTEM BAŞLATILIYOR: TAM KONTROL MODU")

    # 1. ROBOT BAĞLANTISI
    print(f"🤖 Robot ({ROBOT_IP}) aranıyor...")
    try:
        arm = XArmAPI(ROBOT_IP)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        time.sleep(1)
        print("✅ Robot Bağlandı ve Hazır!")
    except Exception as e:
        print(f"❌ Robot Bağlantı Hatası: {e}")
        return

    # Başlangıç pozisyonuna git (Güvenli bölge)
    print("📍 Başlangıç pozisyonuna gidiliyor...")
    arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, speed=50, wait=True)
    
    # Kıskaç (Gripper) Hazırlığı
    arm.set_gripper_enable(True)
    arm.set_gripper_speed(1000)
    arm.set_gripper_position(850, wait=True) # Açık başla

    # 2. YAPAY ZEKA MODELİ
    print("🧠 Yapay Zeka Yükleniyor...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "openvla/openvla-7b"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    print("✅ Beyin Hazır!")

    # 3. KAMERA
    print("👁️ Kamera (Index 0) açılıyor...")
    cap = cv2.VideoCapture(0) # Index 0 olarak güncelledik
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    print("\n⚡ KONTROL BAŞLADI! (Çıkış için 'q', Acil Durum için 'Ctrl+C')")

    try:
        while True:
            # Görüntü Al
            ret, frame = cap.read()
            if not ret: continue

            # AI için Hazırla
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            prompt = "In: What action should the robot take to pick up the object?\nOut:"
            
            # Tahmin
            inputs = processor(prompt, image_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
            
            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # Veriyi Al (7 boyutlu vektör: x, y, z, roll, pitch, yaw, gripper)
            # Model çıktıları çok küçüktür (delta), bunları büyütmeliyiz.
            action_raw = action.flatten()
            
            # --- HAREKET MANTIĞI ---
            # Modelin X'i -> Robotun Y'si (Genellikle kamera açısına göre değişir, deneyerek bulacağız)
            # Modelin Y'si -> Robotun X'i
            
            # Basit Mapping (Kamera robotun tam karşısındaysa):
            # Model  ->  Robot
            # İleri (Y) -> İleri (X)
            # Sağ (X)   -> Sağ (Y) (veya tersi)
            # Yukarı (Z)-> Yukarı (Z)
            
            delta_x = action_raw[0] * SCALE_FACTOR
            delta_y = action_raw[1] * SCALE_FACTOR
            delta_z = action_raw[2] * SCALE_FACTOR
            
            # Mevcut pozisyonu al
            # [x, y, z, roll, pitch, yaw]
            code, current_pose = arm.get_position(is_radian=False)
            if code != 0: continue

            # Yeni hedefleri hesapla
            target_x = current_pose[0] + delta_x
            target_y = current_pose[1] + delta_y # Eğer ters gidiyorsa burayı (-) yap
            target_z = current_pose[2] + delta_z

            # GÜVENLİK SINIRLARI (Masa çarpmasın)
            if target_z < 100: target_z = 100 # Masaya 10cm'den fazla yaklaşma

            # Robota Gönder (Wait=False ile akıcı hareket)
            # Roll, Pitch, Yaw sabit tutuluyor (180, 0, 0) - Sadece XYZ hareketi
            arm.set_position(x=target_x, y=target_y, z=target_z, roll=180, pitch=0, yaw=0, speed=100, wait=False)

            # Gripper Kontrolü
            gripper_val = action_raw[6]
            if gripper_val > GRIPPER_THRESHOLD:
                 # Aç (850)
                 arm.set_gripper_position(850, wait=False)
                 grip_status = "ACIK"
            else:
                 # Kapa (0)
                 arm.set_gripper_position(0, wait=False)
                 grip_status = "KAPALI"

            # Ekrana Bilgi Yaz
            cv2.putText(frame, f"Move: X{delta_x:.1f} Y{delta_y:.1f} Z{delta_z:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Grip: {grip_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Robot Kontrol Ekrani', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 ACİL DURDURMA!")
        arm.set_state(4) # Stop
        arm.disconnect()
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()