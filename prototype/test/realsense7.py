import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import random

# --- 1. AYARLAR ---
# DİKKAT: Segmentasyon için '-seg' uzantılı modeli kullanıyoruz.
# İlk çalıştırmada otomatik indirilecektir.
model = YOLO('models/yolov8n-seg.pt')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Derinlik ölçeğini al
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Hizalama
align_to = rs.stream.color
align = rs.align(align_to)

# --- RENK AYARLARI ---
# COCO datasetindeki 80 sınıf için rastgele ama sabit renkler üretelim.
# Böylece "insan" her zaman aynı renk, "sandalye" farklı bir renk olur.
np.random.seed(42) # Renklerin her çalıştırmada aynı olması için seed sabitliyoruz
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')
# Renkleri BGR formatına uygun liste haline getirelim
colors = [tuple(map(int, color)) for color in colors]


try:
    while True:
        # --- 2. VERİ ALMA VE HAZIRLIK ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rows, cols, _ = color_image.shape

        # Yarı saydam çizim için ana görüntünün bir kopyasını al (Katman)
        overlay = color_image.copy()

        # --- 3. YOLO SEGMENTASYON TESPİTİ ---
        # retina_masks=True daha kaliteli maskeler sağlar
        results = model(color_image, stream=True, verbose=False, retina_masks=True)

        for r in results:
            # Eğer segmentasyon maskesi yoksa bu kareyi atla
            if r.masks is None:
                continue

            # Kutuları ve Maskeleri birlikte döngüye al
            # r.masks.xy -> Maskenin çokgen köşe koordinatlarını verir
            for box, mask_poly in zip(r.boxes, r.masks.xy):
                
                # Sınıf ID'sini ve ona ait rengi al
                cls_id = int(box.cls[0])
                color = colors[cls_id]
                class_name = model.names[cls_id]

                # --- A. SEGMENTASYON ÇİZİMİ ---
                # Çokgen noktalarını OpenCV formatına uygun hale getir (int32)
                contour = mask_poly.astype(np.int32)
                contour = contour.reshape(-1, 1, 2)

                # 1. İçini doldur (Overlay katmanına) - Yarı saydamlık için
                cv2.fillPoly(overlay, [contour], color)
                
                # 2. Dış hattını çiz (Ana resme) - Netlik için
                cv2.drawContours(color_image, [contour], -1, color, 2)

                # --- B. 5x5 ALAN ORTALAMASI HESABI (Mevcut Kodun) ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                kernel_size = 5 
                offset = kernel_size // 2 

                min_x = max(0, cx - offset)
                max_x = min(cols, cx + offset + 1)
                min_y = max(0, cy - offset)
                max_y = min(rows, cy + offset + 1)

                depth_roi = depth_image[min_y:max_y, min_x:max_x]
                valid_pixels = depth_roi[depth_roi > 0]

                if len(valid_pixels) > 0:
                    dist_raw = np.mean(valid_pixels)
                    dist_meters = dist_raw * depth_scale
                    dist_text = f"{dist_meters:.2f}m"
                else:
                    dist_text = "?.??"

                # --- C. ETİKET YAZDIRMA ---
                # Taradığımız 5x5 alanı gösteren küçük kutu (Opsiyonel - Beyaz renk)
                cv2.rectangle(color_image, (min_x, min_y), (max_x, max_y), (255, 255, 255), 1)

                label = f"{class_name} {dist_text}"
                
                # Yazının daha okunaklı olması için arka planına siyah bir kutu ekleyelim
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(color_image, (cx - 5, cy - h - 5), (cx + w + 5, cy + 5), (0,0,0), -1)
                
                # Yazıyı yaz (Beyaz renk)
                cv2.putText(color_image, label, (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- SON AŞAMA: KATMANLARI BİRLEŞTİR ---
        # Overlay (dolgulu) görüntü ile ana görüntüyü karıştır.
        # 0.6 ana resim netliği, 0.4 maske saydamlığı.
        cv2.addWeighted(overlay, 0.4, color_image, 0.6, 0, color_image)

        cv2.imshow('RealSense + YOLOv8 Segmentation + 5x5 Depth', color_image)
        # ESC tuşu (27) ile çıkış
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()