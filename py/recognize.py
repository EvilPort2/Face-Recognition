'This program should be executed third'
import cv2
import sqlite3
from .database import getProfileDataById

def faceRecognize():
    faceCascPath = "haarcascade_frontalface_default.xml"
    eyeCascadePath = "haarcascade_eye.xml"
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    eyeCascade = cv2.CascadeClassifier(eyeCascadePath)

    cam = cv2.VideoCapture(0)
    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('recognized/training.yml')

    id = 0
    while True:
        a, img = cam.read()
        img = cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5, flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(face, 1.1, 5, flags = cv2.CASCADE_SCALE_IMAGE)
            if len(eyes) == 2:
                faceId, confidence = recog.predict(face)
                if confidence < 60:
                    profile = getProfileDataById(str(faceId))
                    name = profile[1]
                    occupation = profile[2]
                    gender = profile[3]
                else:
                    name = "Unknown"
                    occupation = "Unknown"
                    gender = "Unknown"
                cv2.rectangle(img, (x, y), (x + w, y + h), 2)

                cv2.putText(img, "Name- " + name, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cv2.putText(img, "Occupation- " + occupation, (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cv2.putText(img, "Gender- " + gender, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                print ("id = " + str(faceId) + " , confidence = " + str(confidence))
        cv2.imshow("Face Recognition Running", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
