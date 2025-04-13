from ultralytics import YOLO
import time
import torch

VIDEO = "data/wydzial-kopia2.mp4"

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

model = YOLO('models/yolo11n.pt')
model = model.to(mps_device)

start_time = time.time()
results = model.track(source=VIDEO, classes=[0], save=True, save_txt=True, tracker='botsort.yaml', project='results/yolo',show=True, verbose=False)
end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")
