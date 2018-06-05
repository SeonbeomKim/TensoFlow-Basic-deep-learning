import tensorflow as tf #version=1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

train_rate = 0.0001

def train(data):
	batch_size = 256
	loss = 0
	np.random.shuffle(data)

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		train_loss, _ = sess.run([cross_entropy, minimize], {X:input_, Y:target_, keep_prob:0.6})
		loss += train_loss
	
	return loss


def validation(data):
	batch_size = 256
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		vali_loss = sess.run(cross_entropy, {X:input_, Y:target_, keep_prob:1})
		loss += vali_loss
	
	return loss


def test(data):
	batch_size = 256
	correct = 0

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		check = sess.run(correct_check, {X:input_, Y:target_, keep_prob:1})
		correct += check

	return correct / len(data)


def run(train_set, vali_set, test_set):
	for epoch in range(1, 300):
		train_loss = train(train_set)
		vali_loss = validation(vali_set)

		print(epoch)
		if epoch % 10 == 0:
			accuracy = test(test_set)
			print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)

			summary = sess.run(merged, {train_loss_tensorboard:train_loss, vali_loss_tensorboard:vali_loss, test_accuracy_tensorboard:accuracy})
			writer.add_summary(summary, epoch)
		


with tf.device('/gpu:0'):
	X = tf.placeholder(tf.float32, [None, 784]) #batch
	Y = tf.placeholder(tf.float32, [None, 10]) #batch
	keep_prob = tf.placeholder(tf.float32)

	W1 = tf.get_variable('w1', shape = [784, 256], initializer=tf.contrib.layers.xavier_initializer())
	bias1 = tf.Variable(tf.constant(1.0, shape = [256]))
	layer1 = tf.nn.relu(tf.matmul(X, W1) + bias1)
	layer1 = tf.nn.dropout(layer1, keep_prob = keep_prob)

	W2 = tf.get_variable('w2', shape = [256, 256], initializer=tf.contrib.layers.xavier_initializer())
	bias2 = tf.Variable(tf.constant(1.0, shape = [256]))
	layer2 = tf.nn.relu(tf.matmul(layer1, W2) + bias2)
	layer2 = tf.nn.dropout(layer2, keep_prob = keep_prob)

	W3 = tf.get_variable('w3', shape = [256, 256], initializer=tf.contrib.layers.xavier_initializer())
	bias3 = tf.Variable(tf.constant(1.0, shape = [256]))
	layer3 = tf.nn.relu(tf.matmul(layer2, W3) + bias3)
	layer3 = tf.nn.dropout(layer3, keep_prob = keep_prob)

	W4 = tf.get_variable('w4', shape = [256, 10], initializer=tf.contrib.layers.xavier_initializer())
	bias4 = tf.Variable(tf.constant(1.0, shape = [10]))
	output = tf.matmul(layer3, W4) + bias4 
	#pred = tf.nn.softmax(output)

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




#mnist 그림 그리기.
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images)
a = mnist.train.images[5]
b = mnist.train.labels[5]
a = np.reshape(a, (28,28))
print(a, b)

import matplotlib.pyplot as plt #pip install matplotlib
plt.imshow(a) 
plt.title(str(np.argmax(b)))
plt.legend()
plt.show(
'''