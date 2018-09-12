import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

train_rate = 0.001
height = 28
width = 28
channel = 1 #mnist is 1 color
negative_sample = 2
num_classes = 10

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
	import time

	for epoch in range(1, 301):
		start_time = time.time()
		train_loss = train(train_set)
		vali_loss = validation(vali_set)
		accuracy = test(test_set)
		execution_time = time.time() - start_time
		print("epoch : ", epoch, "time: ", execution_time, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)


with tf.name_scope('placeholder'):
	X = tf.placeholder(tf.float32, [None, 784]) #batch
	Y = tf.placeholder(tf.int64, [None, 1]) #batch
	keep_prob = tf.placeholder(tf.float32)
	X_reshape = tf.reshape(X, (-1, height, width, channel))

with tf.name_scope('conv'):
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
	output = tf.layers.dense(flat, units = num_classes, activation=None)

with tf.name_scope('positive_sample'):
	positive_mask = tf.reshape(tf.one_hot(Y, num_classes), [-1, num_classes]) # [N, num_classes]
	positive_mask = positive_mask > 0 # [N, num_classes]
	positive = tf.boolean_mask(output, positive_mask) # [N]
	positive = tf.reshape(positive, [-1, 1]) # [N, 1]

with tf.name_scope('negative_sample'):
	# target(positive)이 sampling 되지 않도록 target 제외하고 전부 True로 index 체크.
	negative_index = tf.one_hot(tf.reshape(Y, [-1]), num_classes) # [N, num_classes]
	negative_index = 1. - negative_index # [N, num_classes]
	negative_index = negative_index > 0 # [N, num_classes]

	# 0~ num_classes를 배치만큼 생성하고 negative_index랑 boolean_mask 처리해서 target(positive) 제외.
	possible_negative_pool = tf.reshape(tf.tile(tf.range(num_classes), [tf.shape(Y)[0]]), (-1, num_classes)) # [N, num_classes]
	negative_index = tf.boolean_mask(possible_negative_pool, negative_index) # [N*(num_classes-1)]
	negative_index = tf.reshape(negative_index, (-1, num_classes-1)) # [N, (num_classes-1)]

	# tensorflow의 shuffle은 가장 겉 단위로만 처리돼서 간단하게라도 셔플 되도록 transpose하고 셔플한 후 다시 transpose 하고, negative 샘플링 할 만큼만 slice
	shuffle = tf.random_shuffle(tf.transpose(negative_index)) # [num_classes-1, N]
	negative_index = tf.transpose(shuffle)[:, :negative_sample] # [N, negative_sample]

	# negative_index로 mask를 생성하고, mask로 output에서 negative value들 꺼내옴.
	negative_mask = tf.one_hot(negative_index, num_classes) # [N, 2, num_classes]
	negative_mask = tf.reduce_sum(negative_mask, axis=1) # [N, num_classes]
	negative_mask = negative_mask > 0 # [N, num_classes]
	negative = tf.boolean_mask(output, negative_mask) # [N*negative_sample]
	negative = tf.reshape(negative, [-1, negative_sample]) # [N, negative_sample]

with tf.name_scope('concat_positive_negative'):
	#positive랑 negative 합쳐서 학습.   positive는 항상 0번 인덱스니까 타겟은 0번인덱스만 1인걸로 생성해서 학습하면 됨.
	sample_logit = tf.concat([positive, negative], axis=-1) # [N, negative_sample+1]
	sample_target = tf.one_hot(tf.tile([0], [tf.shape(Y)[0]]), negative_sample+1) # [N, negative_sample+1]

with tf.name_scope('train'):
	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = sample_target, logits = sample_logit) )	
	optimizer = tf.train.AdamOptimizer(train_rate)
	minimize = optimizer.minimize(cross_entropy)
	correct_check = tf.reduce_sum(tf.cast( tf.equal( tf.argmax(output, 1), tf.reshape(Y, [-1]) ), tf.int32 ))
	
sess = tf.Session()
sess.run(tf.global_variables_initializer())


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
train_set = np.hstack((mnist.train.images, np.reshape(mnist.train.labels, (-1,1)))) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, np.reshape(mnist.validation.labels, (-1, 1))))
test_set = np.hstack((mnist.test.images, np.reshape(mnist.test.labels, (-1, 1))))

run(train_set, vali_set, test_set)
