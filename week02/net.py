import tensorflow as tf

num_epochs = 100
num_batches = 100
depth = 80

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(stape, stddev=0.1)
	return tf.get_variable(initial, name=name)

def conv_layer(inpt, filter_shape, stride, name=None):
	out_channels = filter_shape[3]
	
	_filter = weight_variable(filter_shape)
	conv = tf.nn.conv2d(inpt, filter=_filter, strides=[1, stride, stride, 1], padding='VALID')
	mean, var = tf.nn.moments(conv, axes=[0,1,2])
	beta = tf.get_variable(tf.zeros([out_channels]), name="beta"+name)
	gamma = weight_variable([out_channels], name="gamma"+name)
	
	batch_norm = tf.nn.batch_norm_with_global_normalization( \
		conv, mean, var, beta, gamma, 0.001,
		scale_after_normalization=True)
	
	out = tf.nn.relu(batch_norm)
	
	return out

def f_net(inpt):
	
	conv1 = conv_layer(inpt, [8, 8, 53, 25], 1, 'conv1')
	pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
	conv2 = conv_layer(inpt, [8, 8, 25, 50], 1, 'conv2')
	
	return conv2

def mynet(batchX_155, batchX_67, batchX_23, depth, batch_size):
	#f*f*f
	empty_155 = tf.zeros([batch_size, 155, 155, 50])
	conv1_155 = f_net(tf.concat([batchX_155, empty_155], 3))
	conv2_155 = f_net(tf.concat([batchX_67, conv1_155], 3))
	conv3_155 = f_net(tf.concat([batchX_23, conv2_155], 3))
	out_155 = tf.reshape(conv3_155, [batch_size, 50])
	W_155 = weight_variable([50, depth], name='W_155')
	b_155 = tf.Variable(tf.zeros([batch_size, depth]))
	fc_155 = tf.nn.xw_plus_b(out_155, W_155, b_155)
	#f*f
	empty_67 = tf.zeros([batch_size, 67, 67, 50])
	conv1_67 = f_net(tf.concat([batchX_67, empty_67], 3))
	conv2_67 = f_net(tf.concat([batchX_23, conv1_67], 3))
	out_67 = tf.reshape(conv2_67, [batch_size, 50])
	W_67 = weight_variable([50, depth], name='W_67')
	b_67 = tf.Variable(tf.zeros([batch_size, depth]))
	fc_67 = tf.nn.xw_plus_b(out_67, W_67, b_67)
	#f
	empty_23 = tf.zeros([batch_size, 23, 23, 50])
	conv1_23 = f_net(tf.concat([batchX_23, empty_23], 3))
	out_23 = tf.reshape(conv1_23, [batch_size, 50])
	W_23 = weight_variable([50, depth], name='W_23')
	b_23 = tf.Variable(tf.zeros([batch_size, depth]))
	fc_23 = tf.nn.xw_plus_b(out_23, W_23, b_23)

	out = fc_155 + fc_67 + fc_23

	return out