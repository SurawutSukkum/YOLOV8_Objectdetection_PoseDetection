import os
import random
import math
from ultralytics import YOLO
import cv2
import numpy as np


video_path = 'D:/code_pyLandmark1/MEDIA5.MP4'
video_path = os.path.join(video_path)
video_path_out = '{}_out.mp4'.format(video_path)


cap = cv2.VideoCapture(video_path)

#out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'last.pt')
model_path = os.path.join('.', 'runs', 'segment', 'train', 'weights', 'last.pt')
model_path = 'D:/code_pyLandmark1/yolov8n.pt'
model_path1 = 'D:/code_pyLandmark1/yolov8n-pose.pt'
# Load a model
model = YOLO(model_path)  # load a custom model
model1 = YOLO(model_path1)  # load a custom model

threshold = 0.4

class_name_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

ret, frame1 = cap.read() 
tracker = cv2.legacy_TrackerMOSSE.create()

tracker_id = 0
while True:

    # Capture the entire screen
    #frame1 = ImageGrab.grab()
    ret, frame1 = cap.read() 
    results = model(frame1)[0]
    results1 = model1(frame1)[0]
    frame1 = results1.plot()        

    #cv2.line(frame1, (int(900),int(1000)), (int(1050), int(600)), (0, 0, 255), 5)
        
    for result in results.boxes.data.tolist():
        #print(len(results.boxes.data.tolist()))
        #xyxy, _, confidence, class_id, tracker_id           
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
           cv2.putText(frame1, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
           cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if  int(class_id) == 0:
         #tracker.init(frame1, bbox)        
          cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
          cv2.putText(frame1, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
            
    #success, bbox = tracker.update(frame1)
    #out.write(frame)
    #ret, frame = cap.read()
    cv2.namedWindow('detectLabel', cv2.WINDOW_NORMAL)
    cv2.imshow("detectLabel",frame1)
    if (cv2.waitKey(3) == 27):
      break
        
cap.release()
out.release()
cv2.destroyAllWindows()
