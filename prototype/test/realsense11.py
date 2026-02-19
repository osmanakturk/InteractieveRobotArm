import pyrealsense2 as rs
import numpy as np
import cv2
import time

def safe_full_test():
    pipeline = rs.pipeline()
    config = rs.config()

    # 1. Önce kamerayı bir resetleyelim (Power state hatasını temizlemek için)
    ctx = rs.context()
    if len(ctx.query_devices()) > 0:
        dev = ctx.query_devices()[0]
        dev.hardware_reset()
        print("🔄 Kamera donanımsal olarak resetlendi. 5 saniye bekleniyor...")
        time.sleep(5)

    # 2. Formatları Mac dostu yapalım
    # BGR8 yerine RGB8 bazen daha stabil çalışabilir
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15) # 30 FPS yerine 15
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

    try:
        print("🚀 Yayın başlatılıyor...")
        pipeline.start(config)
        print("✅ Başarılı! Görüntü alınıyor...")

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Numpy'a çevir (RGB8 olduğu için OpenCV'de göstermeden önce BGR'a çevireceğiz)
            color_image = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Ekranda göster
            images = np.hstack((color_image_bgr, depth_colormap))
            cv2.imshow('RealSense Safe Test', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    safe_full_test()