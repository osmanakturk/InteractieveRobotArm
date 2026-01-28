import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import random

# --- 1. AYARLAR ---
# DİKKAT: Segmentasyon modeli '-seg' eki alır.
model = YOLO('./test/yolov8n-seg.pt') 

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

# Her sınıf için rastgele bir renk üretelim (Görsel şölen için)
colors = []
for _ in range(80): # COCO datasetinde 80 sınıf var
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    colors.append((b, g, r))

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rows, cols, _ = color_image.shape

        # Yarı saydam çizim için bir kopya katman oluştur
        overlay = color_image.copy()

        # --- YOLO SEGMENTASYON TESPİTİ ---
        # 'retina_masks=True' daha pürüzsüz maskeler sağlar ama biraz yavaştır.
        # Hız gerekirse False yapabilirsiniz.
        results = model(color_image, stream=True, verbose=False, retina_masks=True)

        for r in results:
            # Eğer hiçbir maske bulunamazsa (sadece kutu bulunduysa) devam et
            if r.masks is None:
                continue
            
            # Maskeleri ve Kutuları eşleştirerek döngüye al
            # r.masks.xy -> Maskenin çokgen koordinatlarını verir
            for mask_poly, box in zip(r.masks.xy, r.boxes):
                
                # --- A. ÇİZİM İŞLEMLERİ (SEGMENTASYON) ---
                cls = int(box.cls[0])
                color = colors[cls] # Sınıfa özel renk seç
                
                # Çokgen noktalarını integer formatına çevir (OpenCV için)
                contour = mask_poly.astype(np.int32)
                contour = contour.reshape(-1, 1, 2) # OpenCV contour formatı

                # 1. İçini yarı saydam doldur (overlay katmanına)
                cv2.fillPoly(overlay, [contour], color)
                
                # 2. Dış hattını çiz (ana resme - daha belirgin olsun)
                cv2.drawContours(color_image, [contour], -1, color, 2)

                # --- B. MESAFE ÖLÇÜMÜ (Önceki 5x5 Kernel Yöntemi) ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                k_size = 5
                offset = k_size // 2
                min_x = max(0, cx - offset)
                max_x = min(cols, cx + offset + 1)
                min_y = max(0, cy - offset)
                max_y = min(rows, cy + offset + 1)

                depth_roi = depth_image[min_y:max_y, min_x:max_x]
                valid_pixels = depth_roi[depth_roi > 0]

                if len(valid_pixels) > 0:
                    dist_m = np.mean(valid_pixels) * depth_scale
                    dist_text = f"{dist_m:.2f}m"
                else:
                    dist_text = "?.??"

                # --- C. ETİKET YAZDIRMA ---
                # Etiketi yine kutunun üzerine yazıyoruz ki okunaklı olsun
                class_name = model.names[cls]
                label = f"{class_name} {dist_text}"
                
                # Etiket arka planı (Daha iyi okunabilirlik için)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(color_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(color_image, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Merkeze nokta koy
                cv2.circle(color_image, (cx, cy), 4, (255, 255, 255), -1)

        # --- SON AŞAMA: YARI SAYDAM KATMANI BİRLEŞTİR ---
        # alpha=0.6 (Ana resim netliği), beta=0.4 (Maske saydamlığı)
        cv2.addWeighted(overlay, 0.4, color_image, 0.6, 0, color_image)

        cv2.imshow('RealSense + YOLOv8 Segmentation', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()