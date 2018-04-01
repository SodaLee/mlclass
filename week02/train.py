import net
import data
import tensorflow as tf
import win_unicode_console
win_unicode_console.enable()

batch_size = 50

def main(restore = False):
	train_set, val_setm, test_set, depth = data.read()
	train_set = train_set.batch(batch_size)
	test_set = test_set.batch(batch_size)

	X_155 = tf.placeholder(tf.float32, [batch_size, 155, 155, 3])
	X_67 = tf.placeholder(tf.float32, [batch_size, 67, 67, 3])
	X_23 = tf.placeholder(tf.float32, [batch_size, 23, 23, 3])
	Y = tf.placeholder(tf.int32, [batch_size])
	label = tf.one_hot(Y, depth)

	rnet = net.mynet(X_155, X_67, X_23, depth, batch_size)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = label, logits = rnet)
	loss_mean = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(1e-3).minimize(loss)
	pred = tf.nn.softmax(rnet)
	top_3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.nn.softmax(rnet), tf.argmax(label, axis = 1), 3), tf.float32))
	
	iterator = train_set.make_initializable_iterator()
	next_ele = iterator.get_next()
	val_iter = val_set.make_initializable_iterator()
	next_val = val_iter.get_next()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("init done")
		for i in range(1000):
			try:
				img_155, img_67, img_23, _y = sess.run(next_ele)
				sess.run(opt, feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y})
				if i % 100 == 0:
					print('iter %5d' % (i))
				print(sess.run(loss_mean, feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y}))
			except tf.errors.OutOfRangeError:
				break
		print("done")

	saver = tf.train.Saver()
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, "./model/model.ckpt")
			print("restored")
		else:
			sess.run(tf.global_variables_initializer())
			print("init done")
		for i in list(range(20)):
			cnt = 0
			acc = 0.0
			lo = 0.0
			sess.run(iterator.initializer)
			while True:
				try:
					img_155, img_67, img_23, _y = sess.run(next_ele)
					dummy, _acc, _lo = sess.run([opt, top_3_acc, loss_mean], feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y})
					acc += _acc
					lo += _lo
					cnt += 1
					if cnt % 25 == 0:
						print(cnt * batch_size, acc / 25, lo / 25)
						acc = 0.0
						lo = 0.0
				except tf.errors.OutOfRangeError:
					break
			print("epoc %d done."%(i+1))

			sess.run(val_iter.initializer)
			cnt = 0
			acc = 0.0
			tacc = 0.0
			lo = 0.0
			while True:
				try:
					img_155, img_67, img_23, _y = sess.run(next_ele)
					_acc, _lo = sess.run([top_3_acc, loss_mean], feed_dict = {X_155: img_155, X_67: img_67, X_23: img_23, Y: _y})
					cnt += 1
					acc += _acc
					tacc += _acc
					lo += _lo
					if cnt % 25 == 0:
						print(cnt * batch_size, acc / 25, lo / 25)
						acc = 0.0
						lo = 0.0
				except tf.errors.OutOfRangeError:
					break
			print("Valid top3 acc:%f"%(tacc / cnt))
			saver.save(sess, "./model/model.ckpt")
			saver.save(sess, "./model/model_%d.ckpt"%i)
			print("model saved")


if __name__ == "__main__":
	main(False)
