import cv2
import numpy as np

boxes_file_path = 'results/frame_2.txt'

TrDict = {
  'csrt': cv2.legacy.TrackerCSRT_create,
  'kcf' : cv2.legacy.TrackerKCF_create,
  'boosting' : cv2.legacy.TrackerBoosting_create,
  'mil': cv2.legacy.TrackerMIL_create,
  'tld': cv2.legacy.TrackerTLD_create,
  'medianflow': cv2.legacy.TrackerMedianFlow_create,
  'mosse':cv2.legacy.TrackerMOSSE_create
  }

v = cv2.VideoCapture('data/uid_vid_00000.mp4')

bounding_boxes  = np.loadtxt(boxes_file_path, dtype=np.float32)
for tracker in TrDict:
  print(f"Using tracker: {tracker}")
  trackers = cv2.legacy.MultiTracker_create()

  v.set(cv2.CAP_PROP_POS_FRAMES, 0)
  ret, frame = v.read()

  for bbi in bounding_boxes:
    print(bbi)
    tracker_i = TrDict[tracker]()
    trackers.add(tracker_i, frame, bbi)

  while True:
    ret, frame = v.read()
    if not ret:
      break
    (success,boxes) = trackers.update(frame)
    for box in boxes:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
v.release()

##################################### KOD Z GITHUBA #####################################
# trackers = cv2.legacy.MultiTracker_create()
# k=10
# for i in range(k):
#     cv2.imshow('Frame',frame)
#     bbi = cv2.selectROI('Frame',frame)
#     tracker_i = TrDict['csrt']()
#     trackers.add(tracker_i,frame,bbi)

# frameNumber = 2
# baseDir = 'results'

# while True:
#     ret, frame = v.read()
#     if not ret:
#         break
#     (success,boxes) = trackers.update(frame)
#     if frameNumber == 2:
#       np.savetxt(baseDir + '/frame_'+str(frameNumber)+'.txt',boxes,fmt='%f')
#     frameNumber+=1
#     for box in boxes:
#         (x,y,w,h) = [int(a) for a in box]
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
#     cv2.imshow('Frame',frame)
#     key = cv2.waitKey(5) & 0xFF
#     if key == ord('q'):
#         break
# v.release()
# cv2.destroyAllWindows()

#########################################################################################