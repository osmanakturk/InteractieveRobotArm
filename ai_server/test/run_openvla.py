import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

def run_vla():
    model_id = "openvla/openvla-7b"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"🚀 Model yükleniyor: {model_id} (Cihaz: {device})")

    try:
        # Processor yüklemesi
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # Hata veren kısım burasıydı. AutoModel yerine AutoModelForVision2Seq kullanıyoruz.
        # OpenVLA bir Vision-to-Action (Sequence) modelidir.
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True # Bu parametre 'configuration_prismatic.py'yi okuması için şart!
        ).to(device)

        print("✅ Model başarıyla yüklendi!")
        
        # Test girdisi
        prompt = "In: What action should the robot take to pick up the object?\nOut:"
        from PIL import Image
        import numpy as np
        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) # Boş test resmi
        
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        
        print("🤖 Tahmin yapılıyor...")
        with torch.no_grad():
            action = model.predict_action(inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"📊 Üretilen Aksiyon: {action}")

    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    run_vla()