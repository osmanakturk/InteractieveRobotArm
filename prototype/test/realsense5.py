import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- 1. AYARLAR ---
model = YOLO('./test/yolov8n.pt')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Akışı başlat
profile = pipeline.start(config)

# --- KRİTİK: KAMERA İÇ PARAMETRELERİNİ (INTRINSICS) AL ---
# 2D pikseli 3D noktaya çevirmek için lensin odak uzaklığını bilmemiz lazım.
# Bu bilgiyi akış başladıktan sonra alabiliriz.
depth_profile = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

# Hizalama
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # --- YOLO TESPİTİ ---
        results = model(color_image, stream=True, verbose=False)

        for r in results:
            for box in r.boxes:
                # Koordinatları al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # --- A. MERKEZ DERİNLİĞİNİ BUL (5x5 KERNEL) ---
                # Boyut hesaplamak için referans derinliğe ihtiyacımız var.
                # Köşelerin derinliğini alırsak arka plana denk gelebilir,
                # bu yüzden cismin merkez derinliğini referans alıyoruz.
                
                rows, cols = depth_image.shape
                # Kernel sınırları
                k_size = 5
                y_min, y_max = max(0, cy - k_size), min(rows, cy + k_size)
                x_min, x_max = max(0, cx - k_size), min(cols, cx + k_size)
                
                depth_roi = depth_image[y_min:y_max, x_min:x_max]
                valid_pixels = depth_roi[depth_roi > 0]

                if len(valid_pixels) == 0:
                    continue # Ölçüm yoksa atla

                # Ortalama derinlik (Metre cinsinden değil, ham veri cinsinden lazım)
                avg_depth_raw = np.mean(valid_pixels)
                dist_meters = avg_depth_raw * depth_scale

                # --- B. BOYUT HESAPLAMA (DEPROJECTION) ---
                # Sol Üst Köşe (Piksel) -> 3D Nokta
                # Not: Derinlik olarak "avg_depth_raw" kullanıyoruz (cismin ön yüzeyi düz kabul edilir)
                point_top_left = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [x1, y1], dist_meters
                )
                
                # Sağ Alt Köşe (Piksel) -> 3D Nokta
                point_bottom_right = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [x2, y2], dist_meters
                )
                
                # point_top_left = [X, Y, Z] koordinatlarıdır (Metre cinsinden)
                
                # Genişlik: X eksenindeki fark
                width_m = abs(point_bottom_right[0] - point_top_left[0])
                # Yükseklik: Y eksenindeki fark
                height_m = abs(point_bottom_right[1] - point_top_left[1])

                # --- ÇİZİM VE YAZI ---
                # Kutuyu çiz
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Bilgileri hazırla
                # cm'ye çevirelim daha okunaklı olsun
                w_cm = width_m * 100
                h_cm = height_m * 100
                d_m = dist_meters
                
                label_dist = f"Dist: {d_m:.2f}m"
                label_size = f"W: {w_cm:.1f}cm H: {h_cm:.1f}cm"
                
                # Ekrana yaz
                cv2.putText(color_image, label_dist, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(color_image, label_size, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 255), 2)

        cv2.imshow('RealSense Boyut Olcum', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()