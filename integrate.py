from ultralytics import YOLO
import cv2
import math 
import numpy as np
import pyttsx3


# model
model = YOLO("yolo-Weights/yolov8n.pt")
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

model.export()