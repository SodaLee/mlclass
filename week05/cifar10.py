import numpy as np
import os
import pickle
data_dir = './'
img_size = 32
num_channels = 3
num_classes = 10
def load_batch(path):
	with open(path, 'rb') as f:
		d = pickle.load(f, encoding='bytes')
		d_decoded = {}
		for k, v in d.items():
			d_decoded[k.decode('utf8')] = v
		d = d_decoded
	data = d['data']
	labels = d['labels']
	data = data.reshape(data.shape[0], 3, 32, 32)
	return data, labels

def load_data():
	dirname = os.path.join(data_dir, 'cifar-10-batches-py/')
	assert os.path.exists(dirname)
	path = dirname
	num_train_samples = 50000
	x_train = np.empty((num_train_samples, 3, 32, 32), dtype = 'uint8')
	y_train = np.empty((num_train_samples,), dtype = 'uint8')
	for i in range(1,6):
		fpath = os.path.join(path, 'data_batch_' + str(i))
		(x_train[(i - 1) * 10000: i * 10000, :, :, :],
		 y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

	x_train = x_train.transpose(0, 2, 3, 1).astype(np.float64)
	#x_train = np.array(x_train).astype(np.uint8)
	y_train = np.array(y_train).astype(np.int64)
	print("train pairs shape:(%s, %s)"%(x_train.shape, y_train.shape))
	return (x_train, y_train)
	
