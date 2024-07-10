from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolov8n.pt")
results = model.track(source=0, conf=0.3, iou=0.5, show=True)