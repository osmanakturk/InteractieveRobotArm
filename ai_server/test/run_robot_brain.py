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

def start_robot_intelligence():
    model_id = "openvla/openvla-7b"
    # M1 Mac için MPS cihazı
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"🧠 Model MPS ({device}) üzerinde başlatılıyor...")
    
    # 1. Model ve Processor Yükleme
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(device)
        print("✅ Model Başarıyla Yüklendi!")
    except Exception as e:
        print(f"❌ Model Yükleme Hatası: {e}")
        return

    # 2. Kamera Hazırlığı (Standart Mod)
    print("👁️ Kamera başlatılıyor...")
    # RealSense genellikle 1. indekstir. Olmazsa 0 veya 2 dene.
    cap = cv2.VideoCapture(0) # <-- Görüntü gelmezse burayı 0 yap
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("⚠️ Kamera 1 açılamadı, 0 deneniyor...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Hiçbir kamera bulunamadı!")
            return

    print("🚀 SİSTEM HAZIR! Robot düşünüyor...")

    try:
        while True:
            # Kameradan kare al
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Kare atlandı")
                time.sleep(0.1)
                continue

            # OpenCV (BGR) -> PIL (RGB) Dönüşümü
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # --- ROBOT BEYNİ (OpenVLA) ---
            prompt = "In: What action should the robot take to pick up the object?\nOut:"
            
            # HATAYI ÇÖZEN KISIM BURASI:
            # Girdileri oluşturuyoruz ama hemen 'to(dtype)' yapmıyoruz
            inputs = processor(prompt, image_pil, return_tensors="pt")
            
            # Girdileri cihaza doğru formatta taşıyoruz
            inputs = inputs.to(device)
            # Sadece resim verisini bfloat16 yapıyoruz (Metinler int kalmalı)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

            # Tahmin
            with torch.no_grad():
                action = model.predict_action(inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # Sonuçları Yazdır
            # Tensor -> Numpy dönüşümü (CPU'ya çekerek)
            action_raw = action
            
            # Terminal Çıktısı
            print(f"🤖 Action: {np.round(action_raw, 3)}")
            
            # Ekranda Göster
            cv2.putText(frame, "AI ANALIZ EDIYOR...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Aksiyon değerlerini ekrana da yazalım
            action_str = str(np.round(action_raw[0][:3], 2)) # Sadece ilk 3 (XYZ) koordinatı
            cv2.putText(frame, f"Move: {action_str}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Robot Gozu (OpenVLA)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Kullanıcı durdurdu.")
    except Exception as e:
        print(f"❌ Kritik Hata: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_robot_intelligence()