import tensorflow as tensorflow
import numpy as np

class Net(object):
	def __init__(self, images, labels):
		self.images = images
		self.labels = labels
		self.mode = mode

	def build_graph(self):
		self.global_step = tf.train.get_or_create_global_step()
		self._build_model()
		if self.mode == 'train':
			self._build_train_op()

	def _build_model(self):
		strides = [1, 2, 2]
		filters = [16, 64, 128, 256]
		x = self.images
		x = self._conv(x, 3, 3, 16, self._stride_arr(1))
		x = self._relu(x)
		for i in range(3):
			x = self._conv(x, 3, filters[i], filters[i+1], strides[i])
			x = self._pool(x)
			x = self._relu(x)

	def _conv(self, x, filter_size, in_filters, out_filters, strides):
		n = filter_size * filter_size * out_filters
		kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32, \
								initializer = tf.random_normal_initializer(stddev = np.sqrt(2.0 / n)))
		return tf.nn.conv2d(x, kernel, strides, padding = 'SAME')

	def _stride_arr(self, stride):
		return [1, stride, stride, 1]

	def _relu(self, x):
		return tf.nn.relu(x)

	def _pool(self, x):
		return tf.nn.max_pool(x, [1, 2, 2, 1], self._stride_arr(2), padding = 'SAME')

