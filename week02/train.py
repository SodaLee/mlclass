import net
import data
import tensorflow as tf

batch_size = 10

def main():
	train_set, test_set, depth = data.read()
	train_set = train_set.batch(batch_size)
	test_set = test_set.batch(batch_size)

	X_155 = tf.placeholder(tf.float32, [batch_size, 155, 155, 3])
	X_67 = tf.placeholder(tf.float32, [batch_size, 67, 67, 3])
	X_23 = tf.placeholder(tf.float32, [batch_size, 23, 23, 3])
	Y = tf.placeholder(tf.int32, [batch_size])

	label = tf.one_hot(Y, depth)
	net = net.mynet(X_155, X_67, X_23, depth, batch_size)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = label, logits = net)
	opt = tf.train.AdamOptimizer(0.02).minimize(loss)
	top_3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.nn.softmax(net), tf.argmax(label, axis = 1), 3), tf.float32))
	
	iterator = train_set.make_one_shot_iterator()
	next_ele = iterator.get_next()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("init done")
		for i in range(10):
			try:
				img_155, img_67, img_23, _y = sess.run(next_ele)
				sess.run(opt, feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y})
				print(sess.run(top_3_acc, feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y}))
			except tf.errors.OutOfRangeError:
				break
		print("done")


if __name__ == "__main__":
	main()
