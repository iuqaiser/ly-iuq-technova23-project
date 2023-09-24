from ultralytics import YOLO
import cv2
import math 
import numpy as np
import pyttsx3

# start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

engine = pyttsx3.init()
engine.say("ad astra abyssosque")
engine.runAndWait()

while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", cls)

            obj_name = r.names[box.cls[0].item()]

            # object details
            if (y1 > cv2.CAP_PROP_FRAME_HEIGHT -10):
                org = [x1, y2]
            else:
                org = [x1, y1] #if box spans to greater than height-10 ig, change orientation to x1 y2
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, obj_name+ " "+ str(int(confidence*100))+"%", org, font, fontScale, color, thickness)
            
            engine.say(obj_name)
            #engine.runAndWait()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
