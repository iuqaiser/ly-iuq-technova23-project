from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyttsx3
import speech_recognition as sr

app = Flask(__name__)

# Initialize the YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 200)

# Initialize speech recognition
recognizer = sr.Recognizer()

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    announce = ''
    announce_width = 0

    engine.say("Hey, are you ready to start?")
    engine.runAndWait()

    end_program = False
    while not end_program:
        audio = listen_to_user_input()
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
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # class name
                    cls = int(box.cls[0])
                    obj_name = r.names[box.cls[0].item()]

                    org = [x1, y2] if y1 > cv2.CAP_PROP_FRAME_HEIGHT - 10 else [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 255, 255)
                    thickness = 2

                    cv2.putText(img, obj_name + " " + str(int(confidence * 100)) + "%", org, font, fontScale, color,
                                thickness)

                    dist = abs(abs((x2 - x1)) - int(width / 2))
                    if (dist >= announce_width):
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

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == 27:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')

            count = count + 1
            count2 = count2 + 1
            if (count >= 20):
                engine.say(announce + " in frame")
                announce_width = 0
                engine.runAndWait()
                count = 0

            if (count2 >= 20):
                engine.say("Average environment color: " + currcolor)
                engine.runAndWait()
                count2 = 0

def listen_to_user_input():
    # getting default microphone instance
    microphone = sr.Microphone()
    with microphone as source:
        # take user input
        print("Listening...")
        audio = recognizer.listen(source)
        recognizer.adjust_for_ambient_noise(source, duration=1)
    return audio

def speech_to_text(audio):
    try:
        text = recognizer.recognize_google(audio)  # uses google web speech api
        text = text.lower()
        print("user input: " + text)
    except sr.UnknownValueError:
        text = ''
        print("Sorry I didn't understand")
    return text

def voice_command_response(text):
    if "yes" in text:
        print("start model")
        return True
    elif "stop" in text:
        print("stop model")
        return False
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
