from ultralytics import FastSAM, YOLO, RTDETR

# Load a model
#model = YOLO("yolov8m-worldv2.engine")
#model = FastSAM("FastSAM-s.engine")
model = RTDETR("rtdetr-l")
results = model.track(source=6, show=True, verbose=False)