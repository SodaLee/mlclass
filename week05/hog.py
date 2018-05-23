from skimage import feature as ft, color
import matplotlib.pyplot as plt
import cifar10
import numpy as np
import random

changed = True

def cal(a, b):
	return np.sqrt(np.sum(np.square(a-b)))

def cal_entropy(a):
	tt = np.zeros(10, np.int64)
	e = np.zeros(10, np.float64)
	for i in range(10):
		for j in range(10):
			tt[j] += a[i][j]
	for i in range(10):
		for j in range(10):
			pij = a[i][j] / tt[j]
			if pij == 0:
				continue
			e[j] -= pij * np.log(pij)
	ret = 0
	for i in range(10):
		ret += tt[i] * e[i]
	return ret / 50000

def k_means(features, labels, centers):
	global changed
	for i, fea in enumerate(features):
		minimum, ind = 1e10, 0
		for j, cen in enumerate(centers):
			dis = cal(fea, cen)
			if dis < minimum:
				minimum = dis
				ind = j + 1
		if labels[i] != ind:
			labels[i] = ind
			changed = True

	centers[:] = np.zeros([10, 36])
	num = np.zeros(10)
	for i, fea in enumerate(features):
		centers[labels[i]-1] += fea
		num[labels[i]-1] += 1
	for i in range(10):
		centers[i] /= num[i]


train_set = cifar10.load_data()

features = np.zeros([50000, 36])
labels = np.zeros(50000, np.uint8)
for i, img_RGB in enumerate(train_set[0]):
	pic = color.rgb2gray(img_RGB)
	features[i] = ft.hog(pic, 9, (16, 16), (2, 2), block_norm='L2-Hys')

indexs = list(range(50000))
random.shuffle(indexs)
centers = np.copy(features[indexs[0:10]])
for i, ind in enumerate(indexs[0:10]):
	labels[ind] = i + 1

while changed:
	changed = False
	cen = np.copy(centers)
	k_means(features, labels, centers)
	print(cal(cen, centers))

print('Evaluating...')
val = np.zeros([10, 10], np.int64)
for i, y in enumerate(train_set[1]):
	val[y][labels[i]-1] += 1
my_table = plt.table(cellText=val, loc='best')
print('Entropy %.4f' % (cal_entropy(val)))
plt.axis('off')
plt.show()
