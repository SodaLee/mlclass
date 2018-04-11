import sys
import models
import tensorflow as tf
import cifar10
from cifar10 import img_size, num_channels, num_classes
import win_unicode_console
win_unicode_console.enable()

batch_size = 50

def main(restore = False, maxiter = 10, test = False):
	train_set, test_set = cifar10.load_data()
	train_set = tf.data.Dataset.from_tensor_slices(train_set).batch(batch_size)
	test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(batch_size)

	X = tf.placeholder(tf.float32, [None, img_size, img_size, num_channels])
	Y = tf.placeholder(tf.int64, [None])
	keep_rate = tf.placeholder(tf.float32)
	
	rnet = models.resnet(X, 32, keep_rate)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y, logits = rnet)
	loss_mean = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(5e-4).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(rnet, 1), Y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#pred = tf.nn.argmax(rnet)
	
	train_iterator = train_set.make_initializable_iterator()
	test_iterator = test_set.make_initializable_iterator()
	next_ele = train_iterator.get_next()
	next_test = test_iterator.get_next()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, "./model/model.ckpt")
			print("restored")
		else:
			sess.run(tf.global_variables_initializer())
			print("init done")
	
			
		for i in list(range(maxiter)):
			cnt = 0
			acc = 0.0
			lo = 0.0
			sess.run(train_iterator.initializer)
			while True:
				try:
					img, _y = sess.run(next_ele)
					dummy, _acc, _lo = sess.run([opt, accuracy, loss_mean], feed_dict = {X: img, Y: _y, keep_rate: 0.5})
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
			
			saver.save(sess, "./model/model.ckpt")
			saver.save(sess, "./model/model_%d.ckpt"%i)
			print("model saved")
			
			acc = 0.0
			cnt = 0
			sess.run(test_iterator.initializer)
			while True:
				try:
					img, labels = sess.run(next_test)
					_acc = sess.run(accuracy, feed_dict = {X: img, Y: labels, keep_rate: 1.0})
					cnt += 1
					acc += _acc
				except tf.errors.OutOfRangeError:
					break
			print("test done, accuracy", acc / cnt, file = sys.stderr)
			
		if test:
			acc = 0.0
			cnt = 0
			sess.run(test_iterator.initializer)
			while True:
				try:
					img, labels = sess.run(next_test)
					_acc = sess.run(accuracy, feed_dict = {X: img, Y: labels, keep_rate: 1.0})
					cnt += 1
					acc += _acc
				except tf.errors.OutOfRangeError:
					break
			print("test done, accuracy", acc / cnt, file = sys.stderr)
		
		

if __name__ == "__main__":
	main(restore=True, maxiter=5, test=False)
	#main(restore=True, maxiter=0, test=True)
