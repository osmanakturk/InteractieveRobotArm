import pyrealsense2 as rs
import time

def force_start():
    print("----------------------------------------")
    print("[1] RealSense Başlatılıyor (Direct Pipeline Modu)...")

    # 1. Pipeline ve Config hazırla
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. Cihaz sorgusu YAPMADAN direkt ayarları gir
    # USB 3.0 ise bu ayarlar çalışır
    try:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    except Exception as e:
        print(f"Config hatası: {e}")

    # 3. Başlatmayı dene
    try:
        print("[2] Pipeline.start() deneniyor...")
        pipeline.start(config)
        print("✅ BAŞARILI! Kamera güç durumu aşıldı ve yayın başladı.")
        
        # 3 saniye yayın yap
        time.sleep(3)
        
        pipeline.stop()
        print("[3] Kapatıldı.")
        
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print("\nÇÖZÜM İPUCU:")
        if "power state" in str(e):
             print("-> Lütfen kamerayı USB'den çıkarıp 5 saniye bekle ve tekrar tak.")
             print("-> RealSense Viewer uygulamasını açıp 'Firmware Update' yap.")

if __name__ == "__main__":
    force_start()