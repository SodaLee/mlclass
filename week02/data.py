import tensorflow as tf

workdir = "./data/"
train_info_path = workdir + "train.info"
test_info_path = workdir + "test.info"

def decode_func(filename, label = None):
	img_string = tf.read_file(workdir+filename)
	img_decode = tf.image.decode_jpeg(img_string)
	img_resized_155 = tf.image.resize_images(img_decode, [155, 155])
	img_resized_67 = tf.image.resize_images(img_decode, [67, 67])
	img_resized_23 = tf.image.resize_images(img_decode, [23, 23])
	img_float_155 = tf.image.convert_image_dtype(img_resized_155, tf.float32)
	img_float_67 = tf.image.convert_image_dtype(img_resized_67, tf.float32)
	img_float_23 = tf.image.convert_image_dtype(img_resized_23, tf.float32)
	if label is not None:
		return img_float_155, img_float_67, img_float_23, label
	else:
		return img_float_155, img_float_67, img_float_23

def read():
	train_filenames = []
	train_labels = []
	test_filenames = []
	depth = 0
	with open(train_info_path, "r") as f:
		for i in f:
			line = i.split()
			train_filenames.append(line[0])
			train_labels.append(int(line[1]))
			depth = max(depth, train_labels[-1])
	with open(test_info_path, "r") as f:
		for i in f:
			test_filenames.append(i)

	train_set = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
	test_set = tf.data.Dataset.from_tensor_slices(test_filenames)

	train_set = train_set.map(decode_func)
	test_set = test_set.map(decode_func)
		
	return train_set, test_set, depth
