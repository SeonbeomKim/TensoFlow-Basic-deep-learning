import numpy as np
import csv

path = './logistic-regression/Social_Network_Ads.csv' # from kaggle 

# 총 400개
#205 학습, 65 검증, 130 테스트.
def preprocess(path):
	data = []
	with open(path, 'r', newline='') as f:
		re = csv.reader(f)
		for line, i in enumerate(re):
			if line != 0: #label 제외
				if i[1] == 'Male': #성멸
					i[1] = 0
				else: #'Female'
					i[1] = 1
				data.append(i)

	data = np.array(data, np.float32)
	data[:, 2] /= np.max(data[:, 2]) #정규화
	data[:, 3] /= np.max(data[:, 3]) #정규화

	np.random.shuffle(data)

	target_0 = [] # 257개 train0 110,  vali0 = 50, test0 = 97 
	target_1 = [] # 143개 train1 95,  vali1 = 15, test1 = 33
	
	for i in range(len(data)): #target이 0인것과 1인것 분리.
		if data[i][4] == 0: 
			target_0.append(data[i])
		else:
			target_1.append(data[i])

	train_set = np.array(target_0[:110]+target_1[:95])
	validation_set = np.array(target_0[110:160]+target_1[95:110])
	test_set = np.array(target_0[160:]+target_1[110:])

	with open("train.csv", 'w', newline='') as o:
		wr = csv.writer(o)
		for i in train_set:
			wr.writerow(i)

	with open("validation.csv", 'w', newline='') as o:
		wr = csv.writer(o)
		for i in validation_set:
			wr.writerow(i)

	with open("test.csv", 'w', newline='') as o:
		wr = csv.writer(o)
		for i in test_set:
			wr.writerow(i)

preprocess(path)