import ObjectCounter
import torch
import json

import sys
sys.path.append('../YOLOv6')

from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.events import load_yaml

from ultralytics import YOLO

SELECTED_CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

with open('coords.json') as f:
    coords = json.load(f)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model_path = '../models/yolo11s.pt'

ObjectCounter = ObjectCounter.ObjectCounter(
    device=device,
    model_path=model_path,
    selected_class_names=SELECTED_CLASS_NAMES,
    frame_stride=3,
    img_size=[1920, 1088],
    half=False,
    coords=coords
)

max_count_videos = ['Br_Koszykowa', 'Br_Niepodleglosci', 'Br_Nowowiejska']

SOURCE_PATH = '/Users/grzegorzsmereczniak/Documents/MyPW/data/monitoring_pw/15_04_25/7-8/Br_Koszykowa/resized/Brama_Koszykowa_192.168.5.149_20250415074005_20250415075242_501697548.mp4'

# TODO: Poprawic zliczanie metoda 'max' - podzielic linie na obszary (jedne wyznaczaja max z jednego obszaru, inne z drugiego i potem suma z tych obszarow)
if __name__ == "__main__":
    for source_path in coords.keys():
        subfolder = source_path.split('/')[-3]
        if subfolder in max_count_videos:
            count_method='max'
        else:
            count_method='sum'
        ObjectCounter.count_objects(
            source_path=source_path,
            count_method=count_method
        )
