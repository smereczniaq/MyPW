import threading
import queue
import cv2
import torch
from ultralytics import YOLO

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device = torch.device("cpu")

# Define model names and video sources
MODEL_NAMES = ["yolo11n.pt", "yolo11x.pt"]
SOURCES = ["GX011022.MP4", "0"]  # local video, 0 for webcam

frame_queue = queue.Queue()


def run_tracker_in_thread(model_name, filename):
    """
    Run YOLO tracker in its own thread for concurrent processing.

    Args:
        model_name (str): The YOLO11 model object.
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
    """
    model = YOLO(model_name).to(mps_device)
    
    # Open video source
    cap = cv2.VideoCapture(filename if filename.isdigit() == False else int(filename))
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {filename}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends
        
        results = model.track(frame, persist=True)  # Perform tracking
        
        # Retrieve annotated frame
        annotated_frame = results[0].plot()

        # Store frame with model name for display
        frame_queue.put((annotated_frame, filename))
    
    cap.release()


# Create and start tracker threads using a for loop
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()

# Wait for all tracker threads to finish
# for thread in tracker_threads:
#     thread.join()

while True:
    if not frame_queue.empty():
        frame, filename = frame_queue.get()
        
        # Show frame with the model name as window title
        cv2.imshow(f"Tracking - {filename}", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()

# Clean up and close windows
cv2.destroyAllWindows()