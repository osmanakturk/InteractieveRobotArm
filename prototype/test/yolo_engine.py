from ultralytics import YOLO, YOLOWorld, RTDETR, SAM, FastSAM

# Load a YOLO26n PyTorch model
model = FastSAM("FastSAM-s.pt")
#model = YOLO("yolo26n.pt")

#result = model.track(source=0, show=True, verbose=False)
# Export the model to TensorRT
model.export(format="engine", device=0, half=True)  # creates 'yolo26n.engine'

# Load the exported TensorRT model
#trt_model = YOLO("yolo26n.engine")

# Run inference
#results = trt_model("https://ultralytics.com/images/bus.jpg")