import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import cv2
import os
import time

# Gereksiz uyarıları kapat
os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

def start_live_intelligence():
    print("🚀 SİSTEM BAŞLATILIYOR: Canlı Yapay Zeka Modu")
    
    # 1. Cihaz Ayarları
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"💻 İşlemci: {device} (Metal Performance Shaders)")

    # 2. Model Yükleme
    model_id = "openvla/openvla-7b"
    print("🧠 Model RAM'e yükleniyor (Lütfen bekleyin)...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(device)
        print("✅ BEYİN HAZIR!")
    except Exception as e:
        print(f"❌ Kritik Model Hatası: {e}")
        return

    # 3. Kamera Başlatma (SDK yerine Standart Driver)
    print("👁️ Kamera bağlanıyor...")
    # RealSense genellikle 1 veya 2 numaradadır. Önce 1'i dene.
    cap = cv2.VideoCapture(3) 
    
    # USB yükünü hafifletmek için çözünürlüğü sabitliyoruz
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("⚠️ Kamera 1 bulunamadı, 0 deneniyor...")
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("❌ HATA: Hiçbir kamera bulunamadı!")
            return

    print("🚀 CANLI AKIŞ BAŞLADI! (Çıkış için 'q' basın)")

    try:
        while True:
            # 1. Kare Yakala
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Kare atlandı...")
                time.sleep(0.1)
                continue

            # 2. Görüntüyü İşle (BGR -> RGB -> PIL)
            # Yapay zeka RGB formatı ister
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # 3. Beyne Gönder (Prompt Hazırla)
            prompt = "In: What action should the robot take to pick up the object?\nOut:"
            
            # --- KRİTİK VERİ İŞLEME (Statik testte doğruladığımız yöntem) ---
            inputs = processor(prompt, image_pil, return_tensors="pt")
            
            # Sözlüğü açıp cihaz ve tip ayarlarını yapıyoruz
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
            
            # 4. Tahmin Al (Aksiyon Vektörü)
            with torch.no_grad():
                # **inputs ile paketi açarak gönderiyoruz (Hata önleyici)
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # 5. Sonucu Görselleştir
            action_raw = action # İlk sonucu al
            
            # Terminale Yazdır
            # Sadece X, Y, Z ve Gripper değerlerini gösterelim (Okunaklı olsun)
            # Format: [X, Y, Z ... Gripper]
            print(f"🤖 Hareket: X={action_raw[0]:.3f}, Y={action_raw[1]:.3f}, Z={action_raw[2]:.3f} | Tutma={action_raw[6]:.1f}")
            
            # Ekrana Yazdır
            text_xyz = f"X: {action_raw[0]:.3f} Y: {action_raw[1]:.3f} Z: {action_raw[2]:.3f}"
            text_grip = f"Gripper: {'AC' if action_raw[6] > 0.5 else 'KAPA'} ({action_raw[6]:.2f})"
            
            # Yeşil kutu ve yazılar
            cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1) # Arkaplan
            cv2.putText(frame, text_xyz, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, text_grip, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('OpenVLA Robot Gozu', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Durduruluyor...")
    except Exception as e:
        print(f"❌ Beklenmedik Hata: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_intelligence()