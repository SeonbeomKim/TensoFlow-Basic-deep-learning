import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

train_rate = 0.001
height = 28
width = 28
channel = 1 #mnist is 1 color

def train(data):
	batch_size = 128
	loss = 0
	np.random.shuffle(data)

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		#print(i+1, '/', int(math.ceil(len(data)/batch_size)))
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		train_loss, _ = sess.run([cross_entropy, minimize], {X:input_, Y:target_, keep_prob:0.6})
		loss += train_loss
	
	return loss


def validation(data):
	batch_size = 512
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		vali_loss = sess.run(cross_entropy, {X:input_, Y:target_, keep_prob:1})
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
		train_loss = train(train_set)
		vali_loss = validation(vali_set)

		#print(epoch)
		#if epoch % 10 == 0:
		accuracy = test(test_set)
		print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)

		summary = sess.run(merged, {train_loss_tensorboard:train_loss, vali_loss_tensorboard:vali_loss, test_accuracy_tensorboard:accuracy})
		writer.add_summary(summary, epoch)
		

with tf.device('/gpu:0'):
	#height = 28
	#width = 28
	#channel = 1
	X = tf.placeholder(tf.float32, [None, 784]) #batch
	Y = tf.placeholder(tf.float32, [None, 10]) #batch
	keep_prob = tf.placeholder(tf.float32)

	X_reshape = tf.reshape(X, (-1, height, width, channel))

	#https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
	layer1 = tf.layers.conv2d(X_reshape, filters=32, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	#https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
	pool_layer1 = tf.layers.max_pooling2d(layer1, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer1 = tf.nn.dropout(pool_layer1, keep_prob = keep_prob)

	layer2 = tf.layers.conv2d(drop_layer1, filters=64, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	pool_layer2 = tf.layers.max_pooling2d(layer2, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer2 = tf.nn.dropout(pool_layer2, keep_prob = keep_prob)

	layer3 = tf.layers.conv2d(drop_layer2, filters=128, kernel_size = [3,3], strides=[1, 1], padding='SAME', activation=tf.nn.relu) #stride = [1, value, value, 1]
	pool_layer3 = tf.layers.max_pooling2d(layer3, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	drop_layer3 = tf.nn.dropout(pool_layer3, keep_prob = keep_prob)

	flat = tf.layers.flatten(drop_layer3) # batch 보존한 상태로 평평하게 펴줌. ==>shape =  batch, 2048
	output = tf.layers.dense(flat, units = 10, activation=None)

	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = output) )	

	optimizer = tf.train.AdamOptimizer(train_rate)
	minimize = optimizer.minimize(cross_entropy)

	correct_check = tf.reduce_sum(tf.cast( tf.equal( tf.argmax(output, 1), tf.argmax(Y, 1) ), tf.int32 ))


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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, mnist.validation.labels))
test_set = np.hstack((mnist.test.images, mnist.test.labels))

run(train_set, vali_set, test_set)

