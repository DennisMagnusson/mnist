#!/usr/bin/python
from __future__ import print_function, division
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 5500

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=1)

class Net:
	def __init__(self):
		sd = 0.1
		with tf.device("/cpu:0"):
			"""
			The default values
			c1 = 24
			c2 = 48
			c3 = 64
			"""
			c1 = 12
			c2 = 24
			c3 = 32

			W1 = tf.Variable(tf.truncated_normal([6, 6, 1,  c1], stddev=sd))
			W2 = tf.Variable(tf.truncated_normal([5, 5, c1, c2], stddev=sd))
			W3 = tf.Variable(tf.truncated_normal([4, 4, c2, c3], stddev=sd))
			W4 = tf.Variable(tf.truncated_normal([7*7*c3, 200],  stddev=sd))
			W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=sd))#Change to 200, 2 in future

			B1 = tf.Variable(tf.constant(0.1, tf.float32, [24]))
			B2 = tf.Variable(tf.constant(0.1, tf.float32, [48]))
			B3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
			B4 = tf.Variable(tf.constant(0.1, tf.float32, [200]))
			B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
		
		#TODO Batch norm this shit
		Y1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
		Y2 = tf.nn.relu(Y1)
		with tf.device("/cpu:0"):#Yeah, something's up with those random values on GPU
			Y3 = tf.nn.dropout(Y2, 0.6)
		Y4 = tf.nn.conv2d(Y3, W2, strides=[1, 2, 2, 1], padding='SAME')
		Y5 = tf.nn.relu(Y4)
		with tf.device("/cpu:0"):
			Y6 = tf.nn.dropout(Y5, 0.6)
		Y7 = tf.nn.conv2d(Y6, W3, strides=[1, 2, 2, 1], padding='SAME')
		Y8 = tf.nn.relu(Y7)
		with tf.device("/cpu:0"):
			Y9 = tf.nn.dropout(Y8, 0.6)
		Y10= tf.reshape(Y9, shape=[-1, 7*7*c3])
		Y11= tf.matmul(Y10, W4) + B4
		Y12= tf.nn.relu(Y11)
		with tf.device("/cpu:0"):
			Y13= tf.nn.dropout(Y12, 0.6)
		Y14= tf.matmul(Y13, W5) + B5
		self.Y = tf.nn.softmax(Y14)



X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_= tf.placeholder(tf.float32, [None, 10])
lr= tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

"""
nets = []
for i in range(10):
	nets.append(Net(i))
"""

net = Net()

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net.Y, labels=y_)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

error = tf.reduce_mean(tf.abs(y_ - net.Y))
correct = tf.equal(tf.argmax(net.Y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#Training
for ep in range(50):
	for i in range(int(math.floor(55000/batch_size))):
		batch_X, batch_Y = mnist.train.next_batch(min(batch_size, 55000-batch_size*i))
		#Learning rate
		lr_max = 0.02
		lr_min = 0.0001
		decay_rate = 1600
		learning_rate = lr_min + (lr_max - lr_min) * math.exp(-ep/decay_rate)
		"""
		#Train the nets in parallel or seperately?
		for n in range(len(nets)):
			lab = convert(batch_Y, n)
			sess.run([train_step, error, nets[n].Y], feed_dict={X: batch_X, y_: lab, lr: learning_rate, training: True})[2]
		"""
			


		#This takes way longer with gpu, but only the first time
		err = sess.run([train_step, error, net.Y], feed_dict={X: batch_X, y_: batch_Y, lr: learning_rate, training: True})[2]
		print(i)

	print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))



def convert(l, k):
	li = []
	for i in l:
		if i[k] == 1:
			li.append([1, 0])
		else:
			li.append([0, 1])

	return np.array(li)
