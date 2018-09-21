import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import time

train_rate = 0.001
height = 28
width = 28
channel = 1 #mnist is 1 color
negative_sample = 1
num_classes = 10

def train(data):
	batch_size = 1024
	loss = 0
	np.random.shuffle(data)

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		#print(i+1, '/', int(math.ceil(len(data)/batch_size)))
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		train_loss, _ = sess.run([sampled_loss, minimize], {X:input_, Y:target_, keep_prob:0.6})
		loss += train_loss
	
	return loss


def validation(data):
	batch_size = 512
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		vali_loss = sess.run(sampled_loss, {X:input_, Y:target_, keep_prob:1})
		loss += vali_loss
	
	return loss


def test(data):
	batch_size = 512
	correct = 0

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		check = sess.run(correct_check, {X:input_, Y:target_, keep_prob:1})
		correct += check

	return correct / len(data)


def run(train_set, vali_set, test_set):
	for epoch in range(1, 301):
		start = time.time()
		train_loss = train(train_set)
		end = time.time()

		vali_loss = validation(vali_set)
		accuracy = test(test_set)
		print("epoch : ", epoch, " time : ", end-start, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)


with tf.name_scope("model"):
	X = tf.placeholder(tf.float32, [None, 784]) #batch
	Y = tf.placeholder(tf.int64, [None, 1]) #batch
	keep_prob = tf.placeholder(tf.float32)
	X_reshape = tf.reshape(X, (-1, height, width, channel))

	layer1 = tf.layers.conv2d(X_reshape, filters=32, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	pool_layer1 = tf.layers.max_pooling2d(layer1, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer1 = tf.nn.dropout(pool_layer1, keep_prob = keep_prob)

	layer2 = tf.layers.conv2d(drop_layer1, filters=64, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	pool_layer2 = tf.layers.max_pooling2d(layer2, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer2 = tf.nn.dropout(pool_layer2, keep_prob = keep_prob)

	layer3 = tf.layers.conv2d(drop_layer2, filters=128, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	pool_layer3 = tf.layers.max_pooling2d(layer3, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer3 = tf.nn.dropout(pool_layer3, keep_prob = keep_prob)

	flat = tf.layers.flatten(drop_layer3) # batch 보존한 상태로 평평하게 펴줌. ==>shape =  batch, 2048

	W = tf.get_variable('W', shape = [2048, num_classes], initializer=tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.constant(0.0, shape=[num_classes]))
	output = tf.matmul(flat, W)+b

	sampled_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(tf.transpose(W), b, Y, flat, negative_sample, num_classes, 1))
	optimizer = tf.train.AdamOptimizer(train_rate)
	minimize = optimizer.minimize(sampled_loss)

	correct_check = tf.reduce_sum(tf.cast( tf.equal( tf.argmax(output, 1), tf.reshape(Y, [-1]) ), tf.int32 ))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
train_set = np.hstack((mnist.train.images, np.reshape(mnist.train.labels, (-1,1)))) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, np.reshape(mnist.validation.labels, (-1, 1))))
test_set = np.hstack((mnist.test.images, np.reshape(mnist.test.labels, (-1, 1))))


run(train_set, vali_set, test_set)
