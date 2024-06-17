import torch
import numpy as np
from ultralytics import YOLO
import cv2
import pafy

# model = YOLO("trained_mask_model.pt") # no mask class = 0
model = YOLO("best.pt") # no mask class = 1
cap = cv2.VideoCapture("mask.mp4")

# python .\YOLO_.py
fps = int(cap.get(cv2.CAP_PROP_FPS)) # 프레임 속도
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 높이
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 동영상 코덱

out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height)) # 출력 동영상 설정

def use_result(frame, results):
  if (results and results[0]) :
    bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
    confidences = np.array(results[0].boxes.conf.cpu(), dtype="float")
    pred_box = zip(classes, bboxes)
    mask = 0
    no_mask = 0
    names = results[0].names
    for cls, bbox, conf in zip(classes, bboxes, confidences):
      (x, y, x2, y2) = bbox
      print("bounding box (",x,y,x2,y2,") has class ", cls, " which is ", names[cls])
      label = f"{names[cls]}: {conf:.2f}"
      if (cls == 1): # no mask class
        cv2.rectangle(frame, (x,y), (x2,y2), (229,142,250), 2)
        cv2.putText(frame, label, (x-50, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (229,142,250), 2)
        cv2.putText(frame, f"Please enter with mask on", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        no_mask += 1
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    cv2.putText(frame, f"No mask count: {no_mask}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # resize image
    frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Img", frame_s)
  out.write(frame)
  return

while True:
  ret, frame = cap.read()
  if not ret:
    break
  results = model(frame)
  use_result(frame, results)

  cv2.imshow("Img", frame)
  key = cv2.waitKey(1)
  # ESC to break
  if key == 27:
    break

out.release()
cap.release()