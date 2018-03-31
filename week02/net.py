import tensorflow as tf

batch_size = 10
num_epochs = 100
num_batches = 100

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

batchX_155 = tf.placeholder(tf.float32, [batch_size, 155, 155, 3])
batchX_67 = tf.placeholder(tf.float32, [batch_size, 67, 67, 3])
batchX_23 = tf.placeholder(tf.float32, [batch_size, 23, 23, 3])
batchY = tf.placeholder(tf.int32, [batch_size, 80])
#f*f*f
empty_155 = tf.zeros([batch_size, 155, 155, 50])
conv1_155 = f_net(tf.concat([batchX_155, empty_155], 3))
conv2_155 = f_net(tf.concat([batchX_67, conv1_155], 3))
conv3_155 = f_net(tf.concat([batchX_23, conv2_155], 3))
out_155 = tf.reshape(conv3_155, [batch_size, 50])
W_155 = weight_variable([50, 80], name='W_155')
b_155 = tf.Variable(tf.zeros([batch_size, 80]))
fc_155 = tf.nn.xw_plus_b(out_155, W_155, b_155)
#f*f
empty_67 = tf.zeros([batch_size, 67, 67, 50])
conv1_67 = f_net(tf.concat([batchX_67, empty_67], 3))
conv2_67 = f_net(tf.concat([batchX_23, conv1_67], 3))
out_67 = tf.reshape(conv2_67, [batch_size, 50])
W_67 = weight_variable([50, 80], name='W_67')
b_67 = tf.Variable(tf.zeros([batch_size, 80]))
fc_67 = tf.nn.xw_plus_b(out_67, W_67, b_67)
#f
empty_23 = tf.zeros([batch_size, 23, 23, 50])
conv1_23 = f_net(tf.concat([batchX_23, empty_23], 3))
out_23 = tf.reshape(conv1_23, [batch_size, 50])
W_23 = weight_variable([50, 80], name='W_23')
b_23 = tf.Variable(tf.zeros([batch_size, 80]))
fc_23 = tf.nn.xw_plus_b(out_23, W_23, b_23)

out = fc_155 + fc_67 + fc_23

predictions_series = [tf.nn.softmax(logits) for logits in out]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=lab) for log, lab in zip(out,batchY)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.02).minimize(total_loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch_idx in range(num_epochs):
		x,y = generateData()
		_current_state = np.zeros((batch_size, state_size))

		print("New data, epoch", epoch_idx)

		for batch_idx in range(num_batches):
			start_idx = batch_idx * truncated_backprop_length
			end_idx = start_idx + truncated_backprop_length

			batchX = x[:,start_idx:end_idx]
			batchY = y[:,start_idx:end_idx]

			_total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

			if batch_idx%100 == 0:
				print("Step",batch_idx, "Loss", _total_loss)
