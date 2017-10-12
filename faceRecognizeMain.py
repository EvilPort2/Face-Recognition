import os
from py.trainer import trainDetector
from py.recognize import faceRecognize
from py.datasetCreator import createDataset
from py.database import displayDbContent

while True:
	if os.name == 'nt':
		os.system("cls")
	else:
		os.system("clear")

	print ("\t\tMAIN MENU\n\t\t---------\n")
	print ("CHOICES:\n\n1. Save a new face or Update an old face\n2. Train detector\n3. Detect and recognize "
		   "faces\n4. Display ID and other information of all the stored faces\n5. Clear "
		   "screen\n6. Exit\n\n")
	while True:
		try:
			choice = int(input("Enter your choice: "))
		except KeyboardInterrupt:
			print("Exiting...")
			exit(1)
		except:
			continue
		if choice >= 1 and choice <= 6:
			break

	if choice == 1:
		createDataset()
		input("Dataset created successfully. Press ENTER to continue...")
	elif choice == 2:
		trainDetector()
		input("Detector is trained and is ready to detect and recognize stored faces. Press ENTER to continue...")
	elif choice == 3:
		print ("Press \'q\' to stop face recognition...")
		faceRecognize()
		input("Press ENTER to continue...")
	elif choice == 4:
		displayDbContent()
		input("Press ENTER to continue...")
	elif choice == 5:
		if os.name != 'nt':
			os.system("clear")
		else:
			os.system("cls")
	else:
		print ("Exiting...")
		exit(1)
