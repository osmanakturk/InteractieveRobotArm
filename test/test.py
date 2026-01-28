from ultralytics import YOLO

# Modeli yükle
model = YOLO('./test/yolov8n-seg.pt')

# Kaynak olarak '0' (webcam) ver ve show=True ile pencereyi aç
# Bu satır kendi döngüsünü içinde çalıştırır.
results = model.track(source=1, show=True, verbose=False)