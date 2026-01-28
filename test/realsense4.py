import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- 1. AYARLAR ---
model = YOLO('./test/yolov8n-seg.pt')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Derinlik ölçeğini al (Depth Scale)
# Ham veriyi (integer) metreye çevirmek için bu çarpan gereklidir.
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Hizalama (Derinlik -> Renk)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # --- 2. VERİ ALMA VE HAZIRLIK ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Görüntüleri Numpy dizisine çevir
        # depth_image, her pikselin milimetre cinsinden uzaklığını tutan bir matristir
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Görüntü boyutlarını al (Sınır kontrolü için)
        rows, cols, _ = color_image.shape

        # --- 3. YOLO TESPİTİ ---
        results = model(color_image, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                class_name = model.names[int(box.cls[0])]

                # --- 4. 5x5 ALAN ORTALAMASI HESABI ---
                # Hedefimiz: Merkez (cx, cy) etrafındaki 5x5'lik kareyi kesip almak.
                # Kernel boyutu (N x N)
                kernel_size = 5 
                offset = kernel_size // 2 # Merkezden ne kadar sola/sağa gideceğiz (2 piksel)

                # Sınırların görüntü dışına taşmamasını sağla (Clamp)
                min_x = max(0, cx - offset)
                max_x = min(cols, cx + offset + 1)
                min_y = max(0, cy - offset)
                max_y = min(rows, cy + offset + 1)

                # İlgili alanı (Region of Interest - ROI) kesip al
                depth_roi = depth_image[min_y:max_y, min_x:max_x]

                # --- KRİTİK NOKTA: SIFIRLARI ELEMEK ---
                # RealSense ölçemediği yerlere 0 verir. Ortalamaya 0'ları katarsak sonuç yanlış çıkar.
                # Sadece 0'dan büyük pikselleri seçiyoruz:
                valid_pixels = depth_roi[depth_roi > 0]

                if len(valid_pixels) > 0:
                    # Geçerli piksellerin ortalamasını al
                    dist_raw = np.mean(valid_pixels)
                    
                    # Ölçek ile çarpıp metreye çevir
                    dist_meters = dist_raw * depth_scale
                    
                    dist_text = f"{dist_meters:.2f}m"
                    color = (0, 255, 0)
                else:
                    # Eğer tüm pikseller 0 ise (Tamamen kör nokta)
                    dist_text = "?.??"
                    color = (0, 0, 255)

                # --- 5. ÇİZİM ---
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                
                # Merkezi ve taradığımız alanı göstermek için küçük bir kutu çizelim (Opsiyonel)
                cv2.rectangle(color_image, (min_x, min_y), (max_x, max_y), (255, 255, 0), 1)

                label = f"{class_name} {dist_text}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                #cv2.rectangle(color_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                #cv2.putText(color_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(color_image, label, (cx+3, cy-1), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('RealSense + YOLO + 5x5 Average', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()