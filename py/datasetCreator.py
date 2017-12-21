'This program should be executed first'
import cv2
import os
import sqlite3
import glob
from .database import getProfileDataById

os.system("clear")
print ("\t\tAdd a new face or Update an old face")
print (44 * "-")


def createDataset():
    faceCascPath = "haarcascade_frontalface_default.xml"
    eyeCascadePath = "haarcascade_eye.xml"
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    eyeCascade = cv2.CascadeClassifier(eyeCascadePath)

    while True:
        try:
            id = int(input("\nEnter the user id: "))
            break
        except KeyboardInterrupt:
            print("Exiting...")
            exit(1)
        except:
            continue
    i = 0
    j = 100

    flag = 1  # if flag is 1 then id does not exist in database else id exists
    conn = sqlite3.connect("faceDetectDatabase.db")
    try:
        cmd = "SELECT * FROM People WHERE ID = " + str(id)
        cursor = conn.execute(cmd)
    except sqlite3.OperationalError:
        cmd1 = "CREATE TABLE People (id varchar(3), name varchar(50), occupation varchar(100), gender varchar(10), lastPictureNumber varchar(10))"
        conn.execute(cmd1)
        cursor = conn.execute(cmd)
    for row in cursor:
        flag = 0

    # ID not in Database
    if flag == 1:
        name = input("Enter new name: ")
        occupation = input("Enter new occupation: ")
        gender = input("Enter new gender: ")
        cmd = "INSERT INTO People VALUES(" + str(id) + ",\"" + str(name) + "\",\"" + str(occupation) + "\",\"" + str(
            gender) + "\",100)"
        conn.execute(cmd)
        conn.commit()
        conn.close()

    # ID is in database
    if flag == 0:
        print ("ID already exists in database. What would you like to do?\n\n1. Delete the old face and add a new face " \
              "with new info to the same ID\n2. " \
              "Update an old face\n ")
        while True:
            try:
                choice = int(input("Choice: "))
            except:
                continue
            if choice >= 1 and choice <= 2:
                break

        if choice == 1:
            cmd = "DELETE FROM People WHERE ID=" + str(id)
            conn.execute(cmd)
            conn.commit()
            name = str(input("Enter new name: "))
            occupation = str(input("Enter new occupation: "))
            gender = str(input("Enter new gender: "))
            for f in glob.glob("dataset/user." + str(id) + "*.jpg"):
                print(f)
                os.remove(f)
            cmd = "INSERT INTO People VALUES(" + str(id) + ",\"" + name + "\",\"" + occupation + "\",\"" + gender + "\",\"" + str(j)+ "\")"
            conn.execute(cmd)
            conn.commit()
            conn.close()

        else:
            profile = getProfileDataById(id)
            i = int(profile[4])
            j = i + 100
            cmd = "UPDATE People SET lastPictureNumber="+str(j)+" WHERE ID="+str(id)
            conn.execute(cmd)
            conn.commit()
            conn.close()

    cam = cv2.VideoCapture(0)
    print(cam.isOpened())
    while cam.isOpened():
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5, flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(face, 1.1, 5, flags = cv2.CASCADE_SCALE_IMAGE)
            if len(eyes) == 2:
                cv2.imwrite("dataset/user." + str(id) + "." + str(i) + ".jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                i = i + 1
            cv2.waitKey(1)
        cv2.imshow("Webcam Face Detection", cv2.flip(img,1))
        cv2.waitKey(1)
        if i > j:
            break

    cam.release()
    cv2.destroyAllWindows()
