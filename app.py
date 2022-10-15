from scipy.spatial import distance as dist
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import requests
import pyrebase

app = Flask(__name__)

firebaseConfig = {"apiKey": "AIzaSyDhye8Jv11RZ-SI8N9K-yAIKP66ql8QFUk",
                  "authDomain": "drowsiness-detection-2cb1a.firebaseapp.com",
                  "databaseURL": "https://drowsiness-detection-2cb1a-default-rtdb.firebaseio.com",
                  "projectId": "drowsiness-detection-2cb1a",
                  "storageBucket": "drowsiness-detection-2cb1a.appspot.com",
                  "messagingSenderId": "160536170519",
                  "appId": "1:160536170519:web:f082ac1749d9ced4568c52",
                  "measurementId": "G-GJFC4QNRJL"}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
# args, _ = vars(ap.parse_args())
args, _ = ap.parse_known_args()

EYE_AR_THRESH = 0.26  # change this to control sleep sensitivity
YAWN_THRESH = 9  # change this to control yawn sensitivity
EYE_AR_CONSEC_FRAMES = 8
YAWN_CONSEC_FRAMES = 8

saying = False
COUNTER2 = 0
COUNTER1 = 0
# global COUNTER2
# global COUNTER1

print("-> Loading the predictor and detector...")
# detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
url = "http://192.168.18.147:8080//video"  # CHANGE IP ADDRESS
# vs = VideoStream(url).start()
# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)


def generate_frames():
    # global alarm_status

    while True:
        ## read the camera frame
        # COUNTER2 = 0
        # COUNTER1 = 0
        success, frame = vs.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # rects = detector(gray, 0)
            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            # distance = lip_distance(shape)
            # print(distance)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER1 += 1

                if COUNTER1 >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # db.child("drowsiness").set("sleep detected")
            else:
                COUNTER1 = 0
                alarm_status = False
                # db.child("drowsiness").set("")
                #
                # if distance > YAWN_THRESH:
                #     COUNTER2 += 1
                #
                #     if COUNTER2 > YAWN_CONSEC_FRAMES:
                #         cv2.putText(frame, "Yawn Alert", (30, 70),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #         # db.child("yawn").set("yawn detected")
                #
                # else:
                #     COUNTER2 = 0
                # # db.child("yawn").set("")
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                # frames = frame.copy()
                frames = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
