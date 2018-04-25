import tensorflow as tf
import numpy as np
import pickle, os

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_number_of_classes():
	return len(os.listdir('faces/'))

num_of_classes = get_number_of_classes()

def mlp_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 128], name="input_layer")
	
	hidden_layer = tf.layers.dense(input_layer, units=128, activation=tf.nn.relu)
	dropout = tf.layers.dropout(hidden_layer, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	hidden_layer = tf.layers.dense(dropout, units=256, activation=tf.nn.relu)
	dropout = tf.layers.dropout(hidden_layer, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	#hidden_layer = tf.layers.dense(dropout, units=1024, activation=tf.nn.relu)
	#dropout = tf.layers.dropout(hidden_layer, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)

	# UNCOMMENT THE LINES BELOW IF THE ACCURACY IS LOW. IT MIGHT HAPPEN IF THERE ARE A HUGE NUMBER OF FACES.
	# hidden_layer = tf.layers.dense(dropout, units=1024, activation=tf.nn.relu)
	# dropout = tf.layers.dropout(hidden_layer, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
	
	logits = tf.layers.dense(dropout, units=num_of_classes, name="output_layer")

	output_class = tf.argmax(input=logits, axis=1, name="output_class")
	output_probab = tf.nn.softmax(logits, name="softmax_tensor")
	predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_of_classes)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='tf_accuracy')}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
	with open("train_features", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("test_features", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)
	#print(len(train_images[1]), len(train_labels))

	classifier = tf.estimator.Estimator(model_fn=mlp_model_fn, model_dir="tmp/mlp_model")

	tensors_to_log = {"probabilities": "softmax_tensor", "classes": "output_class"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_images}, y=train_labels, batch_size=50, num_epochs=400, shuffle=False)
	classifier.train(input_fn=train_input_fn, hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": test_images},
	  y=test_labels,
	  num_epochs=1,
	  shuffle=False)
	test_results = classifier.evaluate(input_fn=eval_input_fn)
	print(test_results)

if __name__ == "__main__":
	tf.app.run()