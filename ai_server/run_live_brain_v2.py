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
    print("🚀 SİSTEM BAŞLATILIYOR: Canlı Yapay Zeka Modu (v2)")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"💻 İşlemci: {device}")

    model_id = "openvla/openvla-7b"
    print("🧠 Model RAM'e yükleniyor...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            #low_cpu_mem_usage=True, 
            trust_remote_code=True
        ).to(device)
        print("✅ BEYİN HAZIR!")
    except Exception as e:
        print(f"❌ Kritik Model Hatası: {e}")
        return

    print("👁️ Kamera bağlanıyor...")
    # RealSense genellikle 1'dedir.
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    
    if not cap.isOpened():
        print("⚠️ Kamera 1 yok, 0 deneniyor...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ HATA: Kamera yok!")
            return

    print("🚀 CANLI AKIŞ BAŞLADI! (Çıkış: 'q')")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Görüntü İşleme
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Prompt
            prompt = "In: What action should the robot take to pick up the object?\nOut:"
            
            # Veri Hazırlama (Fix)
            inputs = processor(prompt, image_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
            
            # Tahmin
            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
            # --- DÜZELTİLEN KISIM (FLATTEN) ---
            # Model çıktısı (1,7) de olsa (7,) de olsa bunu düz bir listeye çevirir.
            # Böylece boyut hatası almayız.
            action_raw = action.flatten()
            
            # Eğer model saçmalayıp boş dönerse diye güvenlik kontrolü
            if len(action_raw) < 7:
                print("⚠️ Model eksik veri üretti, atlanıyor...")
                continue

            # Terminal Çıktısı
            print(f"🤖 Hareket: X={action_raw[0]:.3f}, Y={action_raw[1]:.3f}, Z={action_raw[2]:.3f} | Tutma={action_raw[6]:.1f}")
            
            # Ekranda Gösterim
            # Yeşil kutu
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1) 
            
            # Yazılar
            cv2.putText(frame, "AI BEYIN AKTIF", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Koordinatlar
            coords = f"X:{action_raw[0]:.2f} Y:{action_raw[1]:.2f} Z:{action_raw[2]:.2f}"
            cv2.putText(frame, coords, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Gripper Durumu
            grip_state = "ACIK" if action_raw[6] > 0.5 else "KAPALI"
            grip_color = (0, 255, 0) if action_raw[6] > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"KISKAC: {grip_state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2)
            
            cv2.imshow('OpenVLA Robot Gozu', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Durduruluyor...")
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_intelligence()