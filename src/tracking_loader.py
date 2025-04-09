import cv2
import numpy as np
import json
import time
import os


data_path = 'data'
annotations_path = 'annotation'
results_path = 'results'


TrDict = {
  'csrt': cv2.legacy.TrackerCSRT_create,
  'kcf' : cv2.legacy.TrackerKCF_create,
  'boosting' : cv2.legacy.TrackerBoosting_create,
  'mil': cv2.legacy.TrackerMIL_create,
  'tld': cv2.legacy.TrackerTLD_create,
  'medianflow': cv2.legacy.TrackerMedianFlow_create,
  'mosse':cv2.legacy.TrackerMOSSE_create,
}

tracking_results = {tracker: {} for tracker in TrDict.keys()}


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

        if frame_idx not in gt_data:
            gt_data[frame_idx] = []

        if 'person' in entity["labels"].keys():
            gt_data[frame_idx].append(bbox)

    return gt_data


def get_frames_to_track(gt_data):
    return sorted(gt_data.keys())


def get_objs_to_track(annotation, start_frame):
    objs_to_track = []

    for entity in annotation["entities"]:
        if entity["blob"]["frame_idx"] == start_frame:
            if 'person' in entity["labels"].keys():
                if entity["id"] not in objs_to_track:
                    objs_to_track.append(entity["id"])

    return objs_to_track


def process_video(v, trackers, tracker_type, start_frame, start_bounding_boxes, frames_to_track):
    print(f"Processing video with {tracker_type} tracker...")
    tracker_boxes = {frame_idx: [] for frame_idx in frames_to_track}
    tracker_boxes[start_frame] = start_bounding_boxes

    start_time = time.time()
    v.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = v.read()

    for bbox in start_bounding_boxes:
        tracker_i = TrDict[tracker_type]()
        trackers.add(tracker_i, frame, bbox)

    while True:
        ret, frame = v.read()
        if not ret:
            break

        frame_idx = int(v.get(cv2.CAP_PROP_POS_FRAMES))

        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if frame_idx in frames_to_track:
                tracker_boxes[frame_idx].append((x, y, w, h))

        # cv2.imshow('Frame', frame)
        # key = cv2.waitKey(5) & 0xFF
        # if key == ord('q'):
        #     break

    end_time = time.time()
    elapsed_time = end_time - start_time

    
    return tracker_boxes, elapsed_time


def save_results(tracking_results, video_name, annotation, objs_to_track):
    print(f"Saving results for {video_name}...")
    results = {}

    for tracker_type in tracking_results.keys():
        results[tracker_type] = {
            'entities': [],
            'metadata': annotation['metadata'].copy(),
        }
        results[tracker_type]['metadata']['processing_time'] = tracking_results[tracker_type]['time']

        for frame_idx, boxes in tracking_results[tracker_type]['bounding_boxes'].items():
            for i, box in enumerate(boxes):                
                obj_id = objs_to_track[i]
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
        
                results[tracker_type]['entities'].append(entity)

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
                continue

            annotation = get_annotation(annotation_path)
            gt_data = get_gt_data(annotation)
            frames_to_track = get_frames_to_track(gt_data)
            start_frame = frames_to_track[0]
            objs_to_track = get_objs_to_track(annotation, start_frame)
            start_bounding_boxes = [tuple(box) for box in gt_data[start_frame]]
            tracking_results = {tracker_type: {} for tracker_type in TrDict.keys()}

            for tracker_type in TrDict.keys():
                trackers = cv2.legacy.MultiTracker_create()
                tracking_results[tracker_type]['bounding_boxes'], tracking_results[tracker_type]['time'] = process_video(v, trackers, tracker_type, start_frame, start_bounding_boxes, frames_to_track)

            # cv2.destroyAllWindows()
            v.release()

            save_results(tracking_results, filename, annotation, objs_to_track)
            print(f"Processing time for {filename}: {time.time() - processing_time_start:.2f} seconds")
print("#" * 50)


