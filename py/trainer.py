'This program should be executed second'
import cv2
import os
from PIL import Image
import numpy as np


def getImageAndId(myPath):
    images = [os.path.join(myPath, f) for f in os.listdir(myPath)]
    faces = []
    ID = []
    for image in images:
        faceImg = Image.open(image)
        faceImgNumpyArray = np.array(faceImg, 'uint8')
        id = int(os.path.split(image)[-1].split('.')[1])
        faces.append(faceImgNumpyArray)
        ID.append(id)
        cv2.imshow("Training", faceImgNumpyArray)
        cv2.waitKey(1)
    return faces, ID


def trainDetector():
    recog = cv2.face.LBPHFaceRecognizer_create()
    path = "dataset"
    faces, IDs = getImageAndId("dataset")
    recog.train(faces, np.array(IDs))
    recog.write("recognized/training.yml")
    cv2.destroyAllWindows()
