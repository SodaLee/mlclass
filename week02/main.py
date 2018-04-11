import sys
import data
import models
import tensorflow as tf
import win_unicode_console
win_unicode_console.enable()

batch_size = 50

def main(restore = False, maxiter = 10, test = False):
	train_set, val_set, test_set, depth = data.read(batch_size, 0.0)

	X = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
	Y = tf.placeholder(tf.int32, [batch_size])
	keep_rate = tf.placeholder(tf.float32)
	
	rnet = models.resnet(X, 20, keep_rate)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y, logits = rnet)
	loss_mean = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(5e-4).minimize(loss)
	#opt = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
	pred = tf.nn.softmax(rnet)
	top_3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred, Y, 3), tf.float32))
	pred = tf.nn.top_k(pred, 3)
	
	iterator = train_set.make_initializable_iterator()
	next_ele = iterator.get_next()
	val_iter = val_set.make_initializable_iterator()
	next_val = val_iter.get_next()
	test_iter = test_set.make_one_shot_iterator()
	next_test = test_iter.get_next()

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
			sess.run(iterator.initializer)
			while True:
				try:
					img, _y = sess.run(next_ele)
					dummy, _acc, _lo = sess.run([opt, top_3_acc, loss_mean], feed_dict = {X: img, Y: _y, keep_rate: 0.5})
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
					img, _y = sess.run(next_val)
					_acc, _lo = sess.run([top_3_acc, loss_mean], feed_dict = {X: img, Y: _y, keep_rate: 0.5})
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
			#print("Valid top3 acc:%f"%(tacc / cnt))
			saver.save(sess, "./model/model.ckpt")
			saver.save(sess, "./model/model_%d.ckpt"%i)
			print("model saved")
			
		if test:
			cnt = 0
			while True:
				try:
					img, name = sess.run(next_test)
					val, indices = sess.run(pred, feed_dict = {X: img, keep_rate: 1.0})
					cnt += 1
					for i in range(len(name)):
						print(name[i].decode(), indices[i,0], indices[i,1], indices[i,2])
				except tf.errors.OutOfRangeError:
					break
			print("test done", file = sys.stderr)

if __name__ == "__main__":
	#main(restore=True, maxiter=0, test=False)
	main(restore=True, maxiter=0, test=True)
