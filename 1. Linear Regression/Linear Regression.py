import tensorflow as tf #version=1.4
import csv
import numpy as np
import math
import matplotlib.pyplot as plt #pip install matplotlib

train_path = './random-linear-regression/train.csv' # from kaggle
test_path = './random-linear-regression/test.csv'
train_rate = 0.01


def read_csv(path):
	with open(path, 'r', newline='') as f:
		re = csv.reader(f)

		data = []
		for line, i in enumerate(re):
			if line != 0 and len(i) == 2: #label, 데이터 손상된 라인 있음. train 파일 215번째줄
				data.append(i)
		
	return np.array(data, np.float32)


def train(data):
	batch_size = 32
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, 0]
		target_ = batch[:, 1]
	
		train_loss, _ = sess.run([MSE, minimize], {X:input_, Y:target_})
		loss += train_loss
	
	return loss


def run(data, testset):
	sub = 1
	for epoch in range(1, 17):
		np.random.shuffle(data)
		print(train(data))
		
		plt.subplot(5,4,sub) #train 시각화
		plt.plot(data[:,0], data[:,1], 'ro') # 앞은 color, 뒤 o는 점 #데이터셋
		plt.plot(data[:,0], sess.run(W)*data[:,0]+sess.run(bias), 'b') #직선
		plt.title('after ' + str(epoch) + ' training')
		sub+=1
	
	plt.subplot(5,1,5) #testset 시각화
	plt.plot(testset[:,0], testset[:,1], 'bo') 
	plt.plot(testset[:,0], sess.run(W)*testset[:,0]+sess.run(bias), 'r') 
	plt.title('test ' + str(epoch) + ' W : ' + str(sess.run(W)[0]) + ' bias: ' + str(sess.run(bias)[0]) )

	plt.tight_layout()
	plt.legend()
	plt.show()


X = tf.placeholder(tf.float32, [None]) #batch
Y = tf.placeholder(tf.float32, [None]) #batch

W = tf.Variable(tf.truncated_normal([1]))
bias = tf.Variable(tf.constant(0.0, shape = [1]))

pred = tf.multiply(X, W) + bias
MSE = tf.reduce_mean(tf.square(Y-pred) / 2)

optimizer = tf.train.AdamOptimizer(train_rate)
minimize = optimizer.minimize(MSE)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

trainset = read_csv(train_path)
testset = read_csv(test_path)
run(trainset, testset)