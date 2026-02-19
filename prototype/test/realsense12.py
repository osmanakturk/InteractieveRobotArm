import pyrealsense2 as rs
import numpy as np
import cv2
import time

def surgical_start():
    pipeline = rs.pipeline()
    config = rs.config()

    # 1. Önce formatları en temel seviyeye çekelim
    # Bazı D435i firmwareleri YUYV formatında daha mutlu çalışır
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

    try:
        print("🚀 Pipeline başlatılıyor (Sıralı Deneme)...")
        # Donanıma zaman tanımak için 1 saniye boşluk
        time.sleep(1)
        profile = pipeline.start(config)
        
        # Hizalama nesnesi
        align = rs.align(rs.stream.color)
        print("✅ Kamera uykudan uyandı!")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Mesafe hesapla (Merkez nokta)
            dist = depth_frame.get_distance(320, 240)

            # Görüntü işleme
            color_image = np.asanyarray(color_frame.get_data())
            # YUYV formatında açtığımız için BGR'a çevirmeliyiz
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUYV)
            
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Ekrana Yazdır
            cv2.putText(color_image, f"Mesafe: {round(dist*100, 2)} cm", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(color_image, (320, 240), 5, (0, 0, 255), -1)

            cv2.imshow('RealSense Mesafe Kontrol', np.hstack((color_image, depth_colormap)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Kritik Hata: {e}")
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    surgical_start()