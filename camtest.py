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
model = YOLO("runs/detect/train/weights/best.pt")

engine = pyttsx3.init()
engine.say("ad astra abyssosque")
engine.runAndWait()

"""
# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
"""


while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    text_to_speech = ''

    # coordinates
    for r in results:
        
        boxes = r.boxes
        print("result", boxes)

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
            engine.runAndWait()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

"""
b = img[:, :, :1]
g = img[:, :, 1:2]
r = img[:, :, 2:]
b_mean = np.mean(b)
g_mean = np.mean(g)
r_mean = np.mean(r)
if (b_mean > g_mean and b_mean > r_mean):
    currcolor = "Blue"
elif (g_mean > r_mean and g_mean > b_mean):
    currcolor = "Green"
elif (r_mean > g_mean and r_mean > b_mean):
    currcolor = "Red"
else:
    currcolor = "Unknown"
"""
