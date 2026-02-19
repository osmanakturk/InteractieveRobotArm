import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import os, cv2

# Gereksiz uyarıları kapat
os.environ["USE_TF"] = "NO"
os.environ["USE_JAX"] = "NO"

def test_static_image():
    print("🚀 Başlatılıyor: Final Statik Test")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"💻 Cihaz: {device}")

    model_id = "openvla/openvla-7b"
    
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

    # --- RESİM OLUŞTURMA VE KAYDETME ---
    print("🎨 Test resmi (Kırmızı Kare) oluşturuluyor...")
    image = Image.new('RGB', (224, 224), color=(255, 0, 0))
    
    # Resmi diske kaydet ki sen de görebil
    Image._show(image)
    
    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    
    print("⚙️ Veri işleniyor...")
    inputs = processor(prompt, image, return_tensors="pt")
    
    # Cihaz ayarları
    # inputs bir sözlük (dictionary) olduğu için açıp içindekileri taşıyoruz
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    print("🤖 Model düşünüyor...")
    with torch.no_grad():
        # **inputs kullanarak sözlüğü parametrelere dağıtıyoruz
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # --- DÜZELTME BURADA ---
    # action zaten bir numpy dizisi, tekrar çevirmeye gerek yok!
    print("\n" + "="*40)
    print(f"🎯 SONUÇ (Robot Aksiyon Vektörü):")
    
    # Bilimsel gösterimi (1.2e-5 gibi) kapatıp normal sayı basalım
    with np.printoptions(precision=4, suppress=True):
        print(action)
        
    print("="*40)
    print("✅ TEST TAMAMLANDI!")

if __name__ == "__main__":
    test_static_image()