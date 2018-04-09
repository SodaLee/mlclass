import tensorflow as tf
import random
import numpy as np
import sys

workdir = "./data/"
train_info_path = workdir + "train.info"
test_info_path = workdir + "test.info"

def decode_func(filename, label = None):
	img_string = tf.read_file(workdir+filename)
	img_decode = tf.image.decode_jpeg(img_string)
	img_resized_32 = tf.image.resize_images(img_decode, [224, 224])
	img_float_32 = tf.image.convert_image_dtype(img_resized_32, tf.float32)
	if label is not None:
		return img_float_32, label
	else:
		return img_float_32, filename

def read(batch, validation = 0.1, random_seed = 1145141919):
	train_filenames = []
	train_labels = []
	test_filenames = []
	depth = 0
	random.seed(random_seed)
	with open(train_info_path, "r") as f:
		for i in f:
			line = i.split()
			train_filenames.append(line[0])
			train_labels.append(int(line[1]))
			depth = max(depth, train_labels[-1] + 1)
	with open(test_info_path, "r") as f:
		for i in f:
			line = i.split()
			test_filenames.append(line[0])

	n_val = int(validation * len(train_filenames))
	n_val -= n_val % batch
	n_train = len(train_filenames) - n_val
	n_train -= n_train % batch

	indices = list(range(len(train_filenames)))
	random.shuffle(indices)

	train_filenames = np.array(train_filenames)
	train_labels = np.array(train_labels)

	train_set = tf.data.Dataset.from_tensor_slices((train_filenames[indices[:n_train]], train_labels[indices[:n_train]]))
	val_set = tf.data.Dataset.from_tensor_slices((train_filenames[indices[n_train:n_train+n_val]], train_labels[indices[n_train:n_train+n_val]]))
	test_set = tf.data.Dataset.from_tensor_slices(test_filenames)

	train_set = train_set.map(decode_func)
	train_set = train_set.shuffle(buffer_size = 500)
	train_set = train_set.batch(batch)

	val_set = val_set.map(decode_func)
	val_set = val_set.batch(batch)

	test_set = test_set.map(decode_func)
	test_set = test_set.batch(batch)

	print("read done, n_train: %d n_val: %d n_test: %d, total: %d, label: %d"%
		(n_train, n_val, len(test_filenames), len(test_filenames)+len(train_filenames), depth),
		file = sys.stderr)
		
	return train_set, val_set, test_set, depth
