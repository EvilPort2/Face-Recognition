import csv
import pickle
import numpy as np

DATASET_CSV_FILE = "dataset.csv"
features = []
labels = []
with open(DATASET_CSV_FILE, newline="") as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		labels.append(int(row[0]))
		features.append(np.array(row[1:], dtype=np.float32))

no_of_faces = len(features)
train_features = features[:int(0.9*no_of_faces)]
train_labels = labels[:int(0.9*no_of_faces)]
test_features = features[int(0.9*no_of_faces):]
test_labels = labels[int(0.9*no_of_faces):]

print("Length of train_features", len(train_features))
with open('train_features', 'wb') as f:
	pickle.dump(train_features, f)
del train_features

print("Length of train_labels", len(train_labels))
with open('train_labels', 'wb') as f:
	pickle.dump(train_labels, f)
del train_labels

print("Length of test_features", len(test_features))
with open('test_features', 'wb') as f:
	pickle.dump(test_features, f)
del test_features

print("Length of test_labels", len(test_labels))
with open('test_labels', 'wb') as f:
	pickle.dump(test_labels, f)
del test_labels

