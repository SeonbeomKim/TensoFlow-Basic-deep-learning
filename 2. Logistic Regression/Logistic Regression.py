import tensorflow as tf #version=1.4
import csv
import numpy as np
import math

#after data_preprocess.py
train_path = 'train.csv' #205개
validation_path = 'validation.csv' #65개
test_path = 'test.csv' #130개
train_rate = 0.01

def read_csv(path):
	data = []
	with open(path, 'r', newline='') as f:
		re = csv.reader(f)
		for line, i in enumerate(re):		
				data.append(i)

	return np.array(data, np.float32)


def train(data):
	batch_size = 32
	loss = 0
	np.random.shuffle(data)

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, 1:-1]
		target_ = batch[:, -1]
	
		train_loss, _ = sess.run([cross_entropy, minimize], {X:input_, Y:target_})
		loss += train_loss
	
	return loss


def validation(data):
	batch_size = 32
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, 1:-1]
		target_ = batch[:, -1]
	
		vali_loss = sess.run(cross_entropy, {X:input_, Y:target_})
		loss += vali_loss
	
	return loss


def test(data):
	batch_size = 32
	correct = 0

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, 1:-1]
		target_ = batch[:, -1]

		check = sess.run(correct_check, {X:input_, threshold:0.8, Y:target_})
		correct += check

	return correct / len(data)


def run(train_set, vali_set, test_set):
	for epoch in range(1, 5001):
		train_loss = train(train_set)
		vali_loss = validation(vali_set)
		
		if epoch % 100 == 0:
			accuracy = test(test_set)
			print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)

			summary = sess.run(merged, {train_loss_tensorboard:train_loss, vali_loss_tensorboard:vali_loss, test_accuracy_tensorboard:accuracy})
			writer.add_summary(summary, epoch)


with tf.device('/cpu:0'):
	X = tf.placeholder(tf.float32, [None, 3]) #batch
	Y = tf.placeholder(tf.float32, [None]) #batch

	W = tf.Variable(tf.truncated_normal([3, 1]))
	bias = tf.Variable(tf.constant(1.0, shape = [1]))

	pred = tf.nn.sigmoid(tf.matmul(X, W) + bias) # shape = [batch * 1]
	cross_entropy = -1 * tf.reduce_mean( tf.reshape(Y, [-1, 1]) * tf.log(pred) + tf.reshape((1-Y), [-1, 1]) * tf.log(1-pred) )

	optimizer = tf.train.AdamOptimizer(train_rate)
	minimize = optimizer.minimize(cross_entropy)

	threshold = tf.placeholder(tf.float32)
	cast_pred_by_threshold = tf.cast(tf.greater_equal(tf.reshape(pred, [-1]), threshold), dtype=tf.float32)
	correct_check = tf.reduce_sum(tf.cast(tf.equal(cast_pred_by_threshold, Y), tf.float32))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	#텐서보드 실행 tensorboard --logdir=./tensorboard/. # 띄어쓰기 조심. logdir부터 쭉 다 붙여써야함.
	train_loss_tensorboard = tf.placeholder(tf.float32)
	vali_loss_tensorboard = tf.placeholder(tf.float32)
	test_accuracy_tensorboard = tf.placeholder(tf.float32)

	train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard) 
	vali_summary = tf.summary.scalar("vali_loss", vali_loss_tensorboard)
	test_summary = tf.summary.scalar("test_accuracy", test_accuracy_tensorboard)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./tensorboard', sess.graph)


train_set = read_csv(train_path)
vali_set = read_csv(validation_path)
test_set = read_csv(test_path)

run(train_set, vali_set, test_set)
