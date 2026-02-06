import cv2
from ultralytics import YOLO

# Kendi oluşturduğun TensorRT modelini yükle
model = YOLO("yolo26n-seg.engine")



# Kamerayı başlat (CSI veya USB)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # device=0 diyerek Orin Nano'nun GPU'sunu kullandığından emin oluyoruz
        results = model.track(frame, device=0, stream=True, verbose=False)
        
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Jetson Orin Nano", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()