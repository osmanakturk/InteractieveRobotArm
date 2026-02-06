import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO, FastSAM

# 1. Modeli yükle
#model = YOLO("yolo26x-seg.engine")
model = FastSAM("FastSAM-s.engine")
# 2. RealSense konfigürasyonu
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Akışı başlat
pipeline.start(config)

try:
    while True:
        # Kareleri al
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Görüntüyü numpy dizisine çevir
        frame = np.asanyarray(color_frame.get_data())

        # YOLO tahmini
        results = model.track(frame, device=0, stream=True, verbose=False)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("RealSense YOLOv11", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()