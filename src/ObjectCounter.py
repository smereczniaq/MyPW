import time
import torch
import cv2
import numpy as np
import supervision as sv

from collections import defaultdict
from typing import Callable
from ultralytics import YOLO
from supervision import VideoInfo, VideoSink, Detections
from supervision import VideoInfo, VideoSink, Detections

import sys
sys.path.append('../YOLOv6')

from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.events import load_yaml

CLASS_NAMES_DICT = load_yaml('../datasets/coco.yaml')['names']

class ObjectCounter:
    def __init__(self, device, model_path, selected_class_names, frame_stride, img_size, half, coords):
        self.device = device
        self.model = DetectBackend(model_path, device=self.device) if model_path.split('/')[-1] == 'yolov6s.pt' else YOLO(model_path).to(device)
        self.selected_class_names = selected_class_names
        self.selected_class_ids = [
            {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
            for class_name
            in selected_class_names
        ]

        self.frame_stride = frame_stride
        self.stride = self.model.stride
        self.img_size = img_size
        self.half = half
        self.coords = coords

        self.model.model.float()

    def process_frame(self, frame):
        image = letterbox(frame, self.img_size, self.stride)[0]
        image = image.transpose((2, 0, 1))[::-1].copy()
        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0)
        
        if self.half:
            image = image.half()
        
        image = image.to(self.device)
        return image
    
    def get_results(self, frame):
        with torch.no_grad():
            pred = self.model(frame)
            det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,classes=[0])[0]
        
        xyxy = det[:, :4].cpu().numpy().astype('float32')
        confidence = det[:, 4].cpu().numpy().astype('float32')
        class_id = det[:, 5].cpu().numpy().astype(int)
        mask = None
        tracker_id = None

        return (xyxy, confidence, class_id, tracker_id)
    
    def process_video(
            self, 
            source_path: str, 
            target_path: str, 
            callback: Callable[[np.ndarray, int], np.ndarray]
            ) -> None:
        source_video_info = VideoInfo.from_video_path(video_path=source_path)
        with VideoSink(target_path=target_path, video_info=source_video_info) as sink:
            for index, frame in enumerate(
                sv.get_video_frames_generator(source_path, stride=self.frame_stride)
            ):
                result_frame = callback(frame, index)
                sink.write_frame(frame=result_frame)

    def get_detections(self, xyxy, confidence, class_id, tracker_id):
        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=None,
            tracker_id=tracker_id
        )
    
    def draw_in_out_window(self, frame: np.ndarray, in_counts: dict, out_counts: dict) -> np.ndarray:
        x, y = frame.shape[1] - 300, 30
        width, height = 270, 160
        overlay = frame.copy()

        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 255), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        line_height = 25
        offset_y = y + 25

        for class_name in self.selected_class_names:
            class_id = [k for k, v in CLASS_NAMES_DICT.items() if v == class_name]
            in_count = in_counts[class_id[0]] if class_id else 0
            out_count = out_counts[class_id[0]] if class_id else 0
            cv2.putText(frame, f"{class_name} In: {in_count} Out: {out_count}",
                        (x + 10, offset_y), font, font_scale, font_color, 2)
            offset_y += line_height

        return frame
    
    
    def count_objects(self, source_path, count_method):
        print(f"Processing video: {source_path}")
        video_info = sv.VideoInfo.from_video_path(source_path)
        target_path = source_path.replace('data', 'results')
        coords_list = self.coords[source_path]
        line_coords = []
        line_zones = []
        total_in = defaultdict(int)
        total_out = defaultdict(int)
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
        trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

        for cs in coords_list:
            line_coords.append((sv.Point(*cs[0]), sv.Point(*cs[1])))

        for line_coord in line_coords:
            line_zones.append(sv.LineZone(start=line_coord[0], end=line_coord[1]))

        if count_method == 'max':
            zone_class_counts = [{"in": defaultdict(int), "out": defaultdict(int)} for _ in line_zones]

        line_zone_annotators = [
            sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=0, text_orient_to_line=True)
            for _ in line_zones
        ]

        start_time = time.time()

        # create BYTETracker instance
        byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=video_info.fps,
            minimum_consecutive_frames=3)

        byte_tracker.reset()

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            detections = sv.Detections.from_ultralytics(self.model(frame, conf=0.1, verbose=False)[0]) if isinstance(self.model, YOLO) else self.get_detections(*self.get_results(self.process_frame(frame)))
            detections = detections[np.isin(detections.class_id, self.selected_class_ids)]
            detections = byte_tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for confidence, class_id, tracker_id
                in zip(detections.confidence, detections.class_id, detections.tracker_id)
            ]

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            
            if count_method == 'max':
                for i, (zone, annotator) in enumerate(zip(line_zones, line_zone_annotators)):
                    crossed_in, crossed_out = zone.trigger(detections)
                    annotated_frame = annotator.annotate(annotated_frame, line_counter=zone)

                    for class_id, is_in in zip(detections.class_id, crossed_in):
                        if is_in:
                            zone_class_counts[i]["in"][class_id] += 1

                    for class_id, is_out in zip(detections.class_id, crossed_out):
                        if is_out:
                            zone_class_counts[i]["out"][class_id] += 1

                # compute max across zones
                max_in = defaultdict(int)
                max_out = defaultdict(int)

                for zone_counts in zone_class_counts:
                    for class_id, count in zone_counts["in"].items():
                        max_in[class_id] = max(max_in[class_id], count)
                    for class_id, count in zone_counts["out"].items():
                        max_out[class_id] = max(max_out[class_id], count)

                arg_in = max_in
                arg_out = max_out
            else:
                for zone, annotator in zip(line_zones, line_zone_annotators):
                    crossed_in, crossed_out = zone.trigger(detections)
                    annotated_frame = annotator.annotate(annotated_frame, line_counter=zone)

                    for class_id, is_in in zip(detections.class_id, crossed_in):
                        if is_in:
                            total_in[class_id] += 1

                    for class_id, is_out in zip(detections.class_id, crossed_out):
                        if is_out:
                            total_out[class_id] += 1

                arg_in = total_in
                arg_out = total_out


            # Draw in/out overlay
            annotated_frame = self.draw_in_out_window(annotated_frame, arg_in, arg_out)
            return annotated_frame
        
        self.process_video(
            source_path = source_path,
            target_path = target_path,
            callback=callback
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {target_path}")
        

