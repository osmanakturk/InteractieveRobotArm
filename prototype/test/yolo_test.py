from ultralytics import YOLO

# Modeli yükle
model = YOLO('./test/yolov8s-worldv2.pt')

# Kaynak olarak '0' (webcam) ver ve show=True ile pencereyi aç
# Bu satır kendi döngüsünü içinde çalıştırır.
results = model.track(source=0, show=True, verbose=False)