#!/usr/bin/python
from __future__ import print_function, division
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 1000

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=1)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_= tf.placeholder(tf.float32, [None, 10])
lr= tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

sd = 0.1
with tf.device("/cpu:0"):#Random variables segfaults on gpu
	W1 = tf.Variable(tf.truncated_normal([784, 10], stddev=sd))
	B1 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

a = tf.reshape(X, shape=[-1, 28*28])
Y= tf.nn.softmax(tf.matmul(a, W1) + B1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=y_)
train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

error = tf.reduce_mean(tf.abs(y_ - Y))
correct = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#Training
for ep in range(50):
	for i in range(int(math.floor(55000/batch_size))):
		batch_X, batch_Y = mnist.train.next_batch(min(batch_size, 55000-batch_size*i))

		lr_max = 0.02
		lr_min = 0.0001
		decay_rate = 1600
		learning_rate = lr_min + (lr_max - lr_min) * math.exp(-ep/decay_rate)
		#This takes way longer with gpu, but only the first time
		err = sess.run([train_step, error, Y], feed_dict={X: batch_X, y_: batch_Y, lr: learning_rate, training: True})[2]

	print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))



"""
class Net:
	def __init__(self):
		sd = 0.1
		with tf.device("/cpu:0"):
			W1 = tf.Variable(tf.truncated_normal([784, 10], stddev=sd))
			B1 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

			self.a = tf.reshape(X, shape=[-1, 28*28])
			self.Y = tf.nn.softmax(tf.matmul(self.a, W1) + B1)
			W1 = tf.Variable(tf.truncated_normal([6, 6, 1,  24], stddev=sd))
			W2 = tf.Variable(tf.truncated_normal([5, 5, 24, 48], stddev=sd))
			W3 = tf.Variable(tf.truncated_normal([4, 4, 48, 64], stddev=sd))
			W4 = tf.Variable(tf.truncated_normal([7*7*64, 200],  stddev=sd))
			W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=sd))#Change to 200, 2 in future

			B1 = tf.Variable(tf.constant(0.1, tf.float32, [24]))
			B2 = tf.Variable(tf.constant(0.1, tf.float32, [48]))
			B3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
			B4 = tf.Variable(tf.constant(0.1, tf.float32, [200]))
			B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
		
		#TODO Batch norm
		self.layers = []
		self.layers.append(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'))
		self.layers.append(tf.nn.relu(self.layers[0]))
		self.layers.append(tf.nn.dropout(self.layers[1], 0.6))
		self.layers.append(tf.nn.conv2d(self.layers[2], W2, strides=[1, 2, 2, 1], padding='SAME'))
		self.layers.append(tf.nn.relu(self.layers[3]))
		self.layers.append(tf.nn.dropout(self.layers[4], 0.6))
		self.layers.append(tf.nn.conv2d(self.layers[5], W3, strides=[1, 2, 2, 1], padding='SAME'))
		self.layers.append(tf.nn.relu(self.layers[6]))
		self.layers.append(tf.nn.dropout(self.layers[7], 0.6))
		self.layers.append(tf.reshape(self.layers[8], shape=[-1, 7*7*64]))
		self.layers.append(tf.matmul(self.layers[9], W4) + B4)
		self.layers.append(tf.nn.relu(self.layers[10]))
		self.layers.append(tf.nn.dropout(self.layers[11], 0.6))
		self.layers.append(tf.matmul(self.layers[12], W5) + B5)
		self.Y = tf.nn.softmax(self.layers[13])
"""
