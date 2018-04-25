import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_number_of_classes():
	return len(os.listdir('faces/'))

num_of_classes = get_number_of_classes()

def mlp_model_fn(features):
	input_layer = tf.reshape(features, [-1, 128], name="input_layer")
	
	hidden_layer1 = tf.layers.dense(input_layer, units=128, activation=tf.nn.relu)

	hidden_layer2 = tf.layers.dense(hidden_layer1, units=256, activation=tf.nn.relu)
	
	logits = tf.layers.dense(hidden_layer2, units=num_of_classes)

	return logits
	
def main(argv):
	features = tf.placeholder(tf.float32, [None, 128])
	logits = mlp_model_fn(features)
	output_probab = tf.nn.softmax(logits, name="softmax_tensor")

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		saver.restore(sess, 'tmp/mlp_model/model.ckpt-5067')
		saver.save(sess, 'tmp/mlp_model_inference')

if __name__ == "__main__":
	tf.app.run()
