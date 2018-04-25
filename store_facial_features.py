import cv2
import numpy as np
import dlib
import pickle
import os, csv, shutil
from random import shuffle

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
NEW_FACE_DIR = "new_faces/"
OLD_FACE_DIR = 'faces/'
CSV_FILE = "dataset.csv"

face_rec = dlib.face_recognition_model_v1(MODEL)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
detector = dlib.get_frontal_face_detector()

folder_names = os.listdir(NEW_FACE_DIR)
face_descriptors = []
row = ""
for folder_name in folder_names:
	create_folder(OLD_FACE_DIR+folder_name)
	full_folder_path  = NEW_FACE_DIR+folder_name+"/"
	images = os.listdir(full_folder_path)
	for image in images:
		full_image_path = full_folder_path+image
		print(full_image_path)
		img = cv2.imread(full_image_path)
		cv2.imwrite(OLD_FACE_DIR+folder_name+"/"+image, img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		try:
			face = detector(img, 1)[0]
		except:
			print("Error")
			continue
		shape = shape_predictor(img, face)
		face_descriptor = face_rec.compute_face_descriptor(img, shape)
		face_descriptor = list(face_descriptor)
		face_descriptor.insert(0, int(folder_name))
		with open(CSV_FILE, "a", newline="") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(face_descriptor)
	shutil.rmtree(full_folder_path)
		#print(face_descriptor, type(face_descriptor))
		
with open(CSV_FILE) as f:
	li = f.readlines()

shuffle(li)
shuffle(li)
shuffle(li)
with open(CSV_FILE, 'w') as f:
	f.writelines(li)