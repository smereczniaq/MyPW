VIDEO = 0 #"GX011022.MP4"

from ultralytics import YOLO

import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

model = YOLO('yolo11n.pt')
model = model.to(mps_device)

results = model.track(source=VIDEO, show=True, tracker='bytetrack.yaml')
