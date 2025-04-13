import cv2
import numpy as np
import json
import time
import os
from deep_sort_realtime.deepsort_tracker import DeepSort


data_path = 'data'
annotations_path = 'annotation'
results_path = 'results/deepsort'

tracker = DeepSort(max_age=10)

def get_v(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None
    return cap


def get_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    return annotation


def get_gt_data(annotation):
    gt_data = {}
    for entity in annotation["entities"]:
        frame_idx = entity["blob"]["frame_idx"]
        bbox = entity["bb"]
        object_id = entity["id"]

        if frame_idx not in gt_data:
            gt_data[frame_idx] = []

        if 'person' in entity["labels"].keys():
            gt_data[frame_idx].append({
                "id": object_id,
                "bbox": bbox
            })

    return gt_data


def get_frames_to_track(gt_data):
    return sorted(gt_data.keys())


def process_video(v, tracker, start_frame, start_bounding_boxes, frames_to_track):
    print(f"Processing video with DeepSort...")
    tracker_boxes = {frame_idx: [] for frame_idx in frames_to_track}
    tracker_boxes[start_frame] = start_bounding_boxes
    start_time = time.time()

    detections = []

    for bbox in start_bounding_boxes:
        x, y, w, h = bbox['bbox']
        conf = 1.0
        class_id = 0
        detections.append(([x, y, w, h], conf, class_id))

    while True:
        ret, frame = v.read()
        if not ret:
            break

        active_ids = {}

        frame_idx = int(v.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        tracks = tracker.update_tracks(detections, frame=frame)

        if frame_idx in frames_to_track:
            for i, track in enumerate(tracks):
                if not track.is_confirmed():
                    break

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(float, ltrb)
                w, h = x2 - x1, y2 - y1

                if frame_idx == start_frame:
                    original_id = tracker_boxes[start_frame][i]['id']
                    active_ids[track_id] = original_id
                else:
                    if track_id in active_ids.keys():
                        tracker_boxes[frame_idx].append({
                            'id': active_ids[track_id],
                            'bbox': [x1, y1, w, h],
                        })
                

    end_time = time.time()
    elapsed_time = end_time - start_time

    detections = []
    
    return tracker_boxes, elapsed_time


def save_results(tracking_results, video_name, annotation):
    print(f"Saving results for {video_name}...")
    results = {}

    results = {
        'entities': [],
        'metadata': annotation['metadata'].copy(),
    }
    results['metadata']['processing_time'] = tracking_results['time']

    for frame_idx, boxes in tracking_results['bounding_boxes'].items():
        for box in boxes:                
            obj_id = box['id']
            entity = {
                'bb': box,
                'blob': {
                    'frame_idx': frame_idx,
                },
                'confidence': 1.0,
                'id': obj_id,
                'labels': {
                    'person': 1,
                },
                'time': 1000,
            }
    
            results['entities'].append(entity)

    save_path = os.path.join(results_path, f"{video_name}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {save_path}")
    

for (dirpath, dirnames, filenames) in os.walk(data_path):
    for filename in filenames:
        if filename.endswith('.mp4'):
            processing_time_start = time.time()
            print("#" * 50)
            print(f"Processing {filename}...")
            video_path = os.path.join(data_path, filename)
            annotation_path = os.path.join(annotations_path, f"{filename}.json")
            
            v = get_v(video_path)
            if v is None:
                print(f"Error opening video file: {video_path}")
                continue

            annotation = get_annotation(annotation_path)
            gt_data = get_gt_data(annotation)
            frames_to_track = get_frames_to_track(gt_data)
            start_frame = frames_to_track[0]
            start_bounding_boxes = gt_data[start_frame]
            tracking_results = {}
            tracking_results['bounding_boxes'], tracking_results['time'] = process_video(v, tracker, start_frame, start_bounding_boxes, frames_to_track)
            v.release()

            save_results(tracking_results, filename, annotation)
            print(f"Processing time for {filename}: {time.time() - processing_time_start:.2f} seconds")
print("#" * 50)
