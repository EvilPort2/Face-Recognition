'This program should be executed third'
import cv2
import sqlite3
from .database import getProfileDataById

def faceRecognize():
    faceCascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(faceCascPath)

    cam = cv2.VideoCapture(0)
    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('recognized/training.yml')

    id = 0
    while True:
        a, img = cam.read()
        img = cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), 2)
            id, confidence = recog.predict(gray[y:y + h, x:x + w])
            if confidence < 50:
                profile = getProfileDataById(str(id))
                name = profile[1]
                occupation = profile[2]
                gender = profile[3]
            else:
                name = "Unknown"
                occupation = "Unknown"
                gender = "Unknown"

            cv2.putText(img, "Name- " + name, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(img, "Occupation- " + occupation, (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(img, "Gender- " + gender, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            print ("id = " + str(id) + " , confidence = " + str(confidence))
        cv2.imshow("Face Recognition Running", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
