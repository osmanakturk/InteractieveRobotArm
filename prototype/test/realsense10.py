import pyrealsense2 as rs
import numpy as np
import cv2

def get_live_image():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Mac M1 için en stabil ve düşük yük bindiren ayarlar
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        print("✅ Kamera açıldı! Görüntü için 'q' tuşuna basın.")
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            # Görüntüyü numpy dizisine çevir
            img = np.asanyarray(color_frame.get_data())
            
            cv2.imshow('RealSense Test', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        pipeline.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Kamera Hatası: {e}")

if __name__ == "__main__":
    get_live_image()