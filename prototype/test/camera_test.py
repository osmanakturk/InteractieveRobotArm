import cv2 as cv
import numpy as np
from ultralytics import YOLO, FastSAM, YOLOWorld
import time


def fastsam_test():
    # --- AYARLAR ---
    model_path = "FastSAM-s.engine"  # Engine dosyanın adı  yolov8s-worldv2.engine
    video_path = "/dev/video6"
    
    print(f"[SİSTEM] FastSAM Modeli yükleniyor: {model_path}")
    
    # Modeli yükle
    model = FastSAM(model_path)

    # Kamerayı başlat
    cap = cv.VideoCapture(video_path, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[HATA] Kamera açılmadı: {video_path}")
        exit()

    cv.namedWindow("Jetson FastSAM Engine Test")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        
        # FastSAM Tahmini
        # device=0: GPU kullanır
        # retina_masks=True: Maskelerin daha keskin görünmesini sağlar
        results = model.predict(
            source=frame, 
            device=0, 
            conf=0.4, 
            iou=0.9, 
            retina_masks=True, 
            verbose=False
        )
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # SONUCU ÇİZDİR
        # labels=True: İsimleri yazar
        # boxes=True: Kutuları çizer
        # masks=True: Nesneleri boyar
        annotated_frame = results[0].plot(labels=True, boxes=True, masks=True)

        # FPS bilgisini ekle
        cv.putText(annotated_frame, f"FastSAM FPS: {fps:.1f}", (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Jetson FastSAM Engine Test", annotated_frame)

        if cv.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv.destroyAllWindows()


def yolo_test():
    # --- AYARLAR ---
    # Buraya denemek istediğin .engine dosyasının adını yaz
    model_path = "yolo26s-seg.engine" 
    video_path = "/dev/video6" # RealSense veya USB kamera yolu
    
    print(f"[SİSTEM] Model yükleniyor: {model_path}")
    # Modeli yükle (TensorRT engine otomatik algılanır)
    model = YOLO(model_path)
    #model = FastSAM("FastSAM-s.engine")
    # Kamerayı başlat
    cap = cv.VideoCapture(video_path, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[HATA] Kamera açılmadı: {video_path}")
        exit()

    cv.namedWindow("Jetson YOLO Engine Test")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TAHMİN (Inference)
        # stream=True: Belleği korur, Jetson için daha iyidir.
        # persist=True: Nesne takibi yapar (opsiyonel).
        start_time = time.time()
        results = model.track(frame, persist=True, conf=0.25, device=0, verbose=False)
        end_time = time.time()

        # FPS Hesapla
        fps = 1 / (end_time - start_time)

        # SONUCU ÇİZDİR
        # Ultralytics'in plot() fonksiyonu kutuları ve etiketleri otomatik çizer
        annotated_frame = results[0].plot()

        # Ekrana FPS bilgisini ekle
        cv.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow("Jetson YOLO Engine Test", annotated_frame)

        # 'ESC' (27) veya 'q' ile çıkış
        if cv.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv.destroyAllWindows()


def opencv():

    cv.namedWindow("Camera")



 

    cap = cv.VideoCapture("/dev/video6", cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)

    if not cap.isOpened():
        exit()



    while True:
        ret, frame = cap.read()
        if not ret:
            break


        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__=="__main__":
    fastsam_test()