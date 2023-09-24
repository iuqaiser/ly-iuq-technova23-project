from ultralytics import YOLO
import cv2
import math 
import numpy as np
import pyttsx3
import speech_recognition as sr

# start webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# model
model = YOLO("runs/detect/train/weights/best.pt")

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 200)


recognizer = sr.Recognizer()
# returns user input audio as text
def listen_to_user_input():
    # getting default microphone instance
    microphone = sr.Microphone()
    with microphone as source:
        # take user input
        print("Listening...")
        audio = recognizer.listen(source)
        recognizer.adjust_for_ambient_noise(source, duration=1)
    return audio

# converting the speech to text
def speech_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)  # uses google web speech api
        text = text.lower()
        print("user input: "+text)
    except sr.UnknownValueError:
        text = ''
        print("Sorry I didn't understand")
    return text

# response to command - returns bool
def voice_command_response(text):
    if "yes" in text:
        # run model
        print("start model")
        return True
    elif "stop" in text:
        print("stop model")
        return False

def main():
    count = 0
    count2 = 10
    announce = ''
    announce_width = 0
    

    engine.say("Hey, are you ready to start?")
    engine.runAndWait()

    end_program = False
    while not end_program:
        audio =  listen_to_user_input()
        text = speech_to_text(audio)
        end_program = voice_command_response(text)
        if end_program:
            break
    
    if end_program:

        while True:
            success, img = cap.read()
            results = model(img, stream=True, verbose=False)
            height, width, success = img.shape

            # coordinates
            for r in results:
                
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

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
                    color = (255, 255, 255)
                    thickness = 2

                    cv2.putText(img, obj_name+ " "+ str(int(confidence*100))+"%", org, font, fontScale, color, thickness)

                    dist = abs(abs((x2-x1))-int(width/2))
                    if(dist >= announce_width):
                        announce_width = dist
                        announce = obj_name

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
            
            count= count + 1
            count2 = count2 + 1
            if (count >= 20):        
                engine.say( announce + "in frame")
                announce_width = 0
                engine.runAndWait()
                count = 0
                
            
            if (count2 >= 20):        
                #engine.say(obj_name + "in front of you")
                engine.say("Average environment color: " + currcolor)
                engine.runAndWait()
                count2 = 0
                
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == 27:
                break
            
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("cannot run")



if __name__ == "__main__":
    main()
