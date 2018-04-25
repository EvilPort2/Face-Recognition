import cv2
import numpy as np
import os, time
from webcam_video_stream import WebcamVideoStream
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from threading import Thread
from keras.models import load_model
import pygame, PyHook3, os
import time
import tkinter, win32api, win32con, pywintypes

detector = dlib.get_frontal_face_detector()
FACENET_MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
face_rec = dlib.face_recognition_model_v1(FACENET_MODEL)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
face_recognizer = load_model('mlp_model_keras2.h5')
face_recognizer.predict(np.random.rand(1,128))
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth = 250)

def keyboard_mouse_disable_event(event):
	print("Keyboard and Mouse is disabled")
	return False

def display_lockscreen_text():
	root = tkinter.Tk()
	root.protocol("WM_DELETE_WINDOW", root.quit)
	label = tkinter.Label(root, text='Computer is Locked', font=('Times New Roman','80'), fg='green', bg='white')
	label.master.overrideredirect(True)
	label.master.geometry("+250+250")
	label.master.lift()
	label.master.wm_attributes("-topmost", True)
	label.master.wm_attributes("-disabled", True)
	label.master.wm_attributes("-transparentcolor", "white")
	hWindow = pywintypes.HANDLE(int(label.master.frame(), 16))
	exStyle = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT
	win32api.SetWindowLong(hWindow, win32con.GWL_EXSTYLE, exStyle)
	label.pack()
	while True:
		root.update()
		try:
			with open("match_face_result2", 'rb') as f:
				matched = f.read()
			if matched:
				os.remove('match_face_result2')
				break
		except:
			continue
		if matched:
			break
	root.quit()

def lock_keyboard_mouse():
	hm = PyHook3.HookManager()
	hm.KeyAll = keyboard_mouse_disable_event
	hm.MouseAll = keyboard_mouse_disable_event
	hm.HookKeyboard()
	hm.HookMouse()
	pygame.init()
	while True:
		pygame.event.pump()
		try:
			with open("match_face_result1", 'r') as f:
				matched = bool(f.read())
			if matched:
				os.remove('match_face_result1')
				break
			
		except:
			time.sleep(0.01)
			continue

def get_unlock_faceid():
	while True:
		unlock_faceid = input("Enter the face id to unlock: ")
		try:
			unlock_faceid = int(unlock_faceid)
			break
		except:
			print("Enter an integer")
			continue
	return unlock_faceid

def recognize_face(face_descriptor):
	#print(face_descriptor)
	pred = face_recognizer.predict(face_descriptor)
	pred_probab = pred[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def extract_face_info(img, img_rgb):
	faces = detector(img_rgb)
	x, y, w, h = 0, 0, 0, 0
	face_descriptor = None
	if len(faces) == 1:
		face = faces[0]
		shape = shape_predictor(img, face)
		(x, y, w, h) = face_utils.rect_to_bb(face)
		face_descriptor = face_rec.compute_face_descriptor(img_rgb, shape)
		face_descriptor = np.array([face_descriptor,])	
	return x, y, w, h, face_descriptor

def start_unlocker(unlock_faceid):
	vs = WebcamVideoStream(1).start()
	x, y, w, h = 0, 0, 0, 0
	match_percentage = 0
	match_frames = 0
	total_frames = 50
	start_time = time.time()
	face_recognizer.predict(np.random.rand(1,128))
	while True:
		img = vs.read()
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		x, y, w, h, face_descriptor = extract_face_info(img, img_rgb)
		if np.any(face_descriptor != None):
			probability, face_id = recognize_face(face_descriptor)
			print(probability, face_id)
			if probability > 0.5:
				cv2.putText(img, "Face #"+str(face_id), (x, y - 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(img, "%s %.2f%%" % ('Probability', probability*100), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				if int(face_id) == unlock_faceid:
						match_frames += 1
				else:
					match_frames = 0
			else:
				cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

			match_percentage = (match_frames/total_frames) * 100
			if match_percentage <= 10:
				cv2.putText(img, "Matching... " + str(match_percentage) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)		#red
			elif match_percentage <= 20:
				cv2.putText(img, "Matching... " + str(match_percentage) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 165, 255), 2)	#orange
			elif match_percentage <= 60:
				cv2.putText(img, "Matching... " + str(match_percentage) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)	#yellow
			elif match_percentage <= 99:
				cv2.putText(img, "Matching... " + str(match_percentage) + "%", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 205, 154), 2) #lime green
			if match_percentage > 100:
				with open('match_face_result1', 'w') as f:
					f.write('True')
				with open('match_face_result2', 'w') as f:
					f.write('True')
				break
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

		cv2.imshow('Recognizing faces', img)
		cv2.waitKey(1)

	vs.stop()
	cv2.destroyAllWindows()


unlock_faceid = get_unlock_faceid()
Thread(target=display_lockscreen_text).start()
Thread(target=lock_keyboard_mouse).start()
start_unlocker(unlock_faceid)