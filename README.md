## Requirements
1. Python 3.x
2. Tensorflow
3. OpenCV
4. Dlib
5. numpy
6. Keras
7. imutils

## How to use the files (stepwise)

### Save few faces first

1. Run the save_face.py file

		python save_face.py

2. It will ask for face_id. Since I have already saved 8 different faces of myself and 7 other friends, your face_id should start from 9.
3. It will also ask for a starting image number. Enter it as 1. If you have already done this and you still want to add more images then enter this as 301. Then for even more 601 and so on.
4. Now a window showing your webcam feed should appear. Make sure there is only one face in the frame or else the face capturing will stop. Also make sure that you give facial expressions during the capturing. After 300 images of your face are taken the window automatically stops.
5. You can see the faces saved in the new_faces/&lt;face_id&gt; directory.
6. In the faces directory you will see some subfolders named as '0', '1', '2' etc. These numbers represent the face_id. Inside each folder you will see 300 images of the person taken from the webcam.
7. You can add your own images that are taken from your phone or any other device inside the new_faces/&lt;face_id&gt; depending on your face_id.

### Storing 128 facial measurements aka embeddings in a csv file

1. Run the store_facial_features.py file
	
		python store_facial_features.py

2. What this file does is it iteratively searches for every face in the faces folder, then computes the embeddings of each of them.
3. These embeddings are stored in a csv file called dataset.csv.
4. The format of the each row of csv file is <b>face_id</b>, <b>facial measurement 1</b>, <b>facial measurement 2</b>,....,<b>facial measurement 128</b>.

### Getting training and testing data from the csv file

1. Run the csv_to_pickle.py file
	
		python csv_to_pickle.py

2. The top 9/10th of the data in the csv file is used as training data and the rest 1/10th is used as testing data.
3. 4 new files will be created train_features, train_labels, test_features and test_labels.

### Train the model

1. Run the train_model_tf.py file to train using Tensorflow. Remember to delete the tmp/ folder first if other faces are added or else you will face errors with shape of the output data.

		python train_model_tf.py

2. Run the train_model_keras.py file to train using Keras

		python train_model_keras.py

3. The emdbedding are used to train a multilayer perceptron.
4. I used my rule of thumb to create this network. I have no idea why it works, but I know it works.
5. With 4 different faces, I got 100% accuracy on the validation data using this network.
6. Add many more faces to see if the network really works.
7. The network can be used in mobile devices too since the network is very simple.
8. The checkpoint files are stored in the tmp/mlp_model/ folder.
9. mlp_model_keras2.h5 file will be created if you use the train_model_keras.py

### Recognition

1. Make sure to run the train_model_keras.py file first.
2. Run the recognition.py file
		
		python recognize.py

### Lockscreen (Experimantal)

1. If you want to unlock your computer using your face run the lockscreen.py
	
		python lockscreen.py
2. You won't be able to use the keyboard and mouse when the computer is locked.
3. Also a green text saying 'Computer is locked' will be displayed on the screen
4. Highly experimantal. In case of a crash hit Ctrl+Alt+Delete

### Generating the Intel Movidius NCS graph

1. Generating the NCS graph is easy. The graph file generated (if the following steps are followed) is stored in the 'NCS graph' folder.
2. Make sure you have trained the network by running train_model_tf.py file.
3. Create the same neural network only for inference and not for training. So remove all the parts that are related to training like dropout layers, loss, optimizers etc. Also make sure you name the input and the output layer. I did that and stored the code in infer_model_tf.py file. You can run this file using
		
		python infer_model_tf.py

4. Running the above file will take only a second since we are not training. This creates a model that is NCS friendly.
5. You will find some new files in the tmp/ folder namely <b>mlp_model_inference.index</b>, <b>mlp_model_inference.meta</b> and <b>mlp_model_inference.data-00000-of-00001</b>.
6. With the NCS friendly model created you can now create the graph file with this command (make sure you are in the face-recognition1 folder)

		mvNCCompile tmp/mlp_model_inference.meta -in input_layer -on softmax_tensor -o NCS\ graph/mlp_model.graph
		
7. This will create the required graph file in the 'NCS graph' folder.

## Future Work

1. Recognition using the NCS SDK.
