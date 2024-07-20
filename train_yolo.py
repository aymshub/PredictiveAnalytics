from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data="dataset.yaml", epochs=50, imgsz=360)

# Uncomment these lines if you encounter issues with labels or need to train further with different configurations
# results = model.train(data="dataset-no-rotation.yaml", epochs=100, imgsz=640)
# results = model.train(data="dataset-rotated.yaml", epochs=100, imgsz=640)
