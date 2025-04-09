from ultralytics import RTDETR
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-x.pt").to(mps_device)

# Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model(source="GX011022.MP4", show=True)