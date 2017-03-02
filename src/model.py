from __future__ import print_function

import tensorflow as tf
import numpy as np

def weightVariable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	return tf.Variable(initial)

def biasVariable(shape):
	initial = tf.constant(0.1, shape=shape, name='biases')
	return tf.Variable(initial)

def conv2d(x, W, pad='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=pad)

def maxPool2x2(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=pool_name)

def maxPool2x1(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME', name=pool_name)

def biLSTM(x, nInputs, nHidden, sco="bidirectional_rnn"):
	lstmFwCell = tf.contrib.rnn.BasicLSTMCell(nHidden, forget_bias=1.0)
	lstmBwCell = tf.contrib.rnn.BasicLSTMCell(nHidden, forget_bias=1.0)
	try:
		outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstmFwCell, lstmBwCell, x, dtype=tf.float32, scope=sco)
	except Exception:
		outputs = tf.contrib.rnn.static_bidirectional_rnn(lstmFwCell, lstmBwCell, x, dtype=tf.float32, scope=sco)
	return outputs

class CRNN:
	def __init__(self, inputImgs, conf, isTraining, weights=None, session=None):
		self.inputImgs = inputImgs
		self.sess = session
		self.convLayers()
		self.lstmLayers()
		self.prob = self.softmax
		self.config = conf
		self.isTrain = isTraining
		if self.sess is not None and weights is not None:
			loadWeights(weights)
	def convLayers(self):
		#preprocess
		with tf.name_scope('preprocess') as scope:
			#images = tf.reshape(self.inputImgs, [-1, 32, 100, 1])
			#addConst = tf.constant(-128.0, shape=images.shape.as_list())
			#mulConst = tf.constant(1.0 / 128, shape=images.shape.as_list())
			#images = tf.add(images, addConst)
			#self.imgs = tf.multiply(images, mulConst, name=scope)
			images = self.inputImgs
			images = np.add(images, -128.0)
			images = np.multiply(images, 1.0/128.0)
			self.imgs = images
		print(self.imgs.shape)
		#conv1
		with tf.name_scope('conv1') as scope:
			kernel = weightVariable([3, 3, 1, 64])
			conv = conv2d(self.imgs, kernel)
			biases = biasVariable([64])
			out = tf.nn.bias_add(conv, biases)
			self.conv1 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool1
		print(self.conv1.shape)
		self.pool1 = maxPool2x2(self.conv1, 'pool1')
		print(self.pool1.shape)
		#conv2
		with tf.name_scope('conv2') as scope:
			kernel = weightVariable([3, 3, 64, 128])
			conv = conv2d(self.pool1, kernel)
			biases = biasVariable([128])
			out = tf.nn.bias_add(conv, biases)
			self.conv2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool2
		print(self.conv2.shape)
		self.pool2 = maxPool2x2(self.conv2, 'pool2')
		print(self.pool2.shape)
		#conv3_1 w/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv3_1') as scope:
			kernel = weightVariable([3, 3, 128, 256])
			conv = conv2d(self.pool2, kernel)
			biases = biasVariable([256])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=True)
			self.conv3_1 = tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		print(self.conv3_1.shape)
		#conv3_2
		with tf.name_scope('conv3_2') as scope:
			kernel = weightVariable([3, 3, 256, 256])
			conv = conv2d(self.conv3_1, kernel)
			biases = biasVariable([256])
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		print(self.conv3_2.shape)
		#maxPool3
		self.pool3 = maxPool2x1(self.conv3_2, 'pool3')
		print(self.pool3.shape)
		#conv4_1 w/batch_norm
		with tf.name_scope('conv4_1') as scope:
			kernel = weightVariable([3, 3, 256, 512])
			conv = conv2d(self.pool3, kernel)
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=True)
			self.conv4_1 = tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		print(self.conv4_1.shape)
		#conv4_2 wo/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv4_2') as scope:
			kernel = weightVariable([3, 3, 512, 512])
			conv = conv2d(self.conv4_1, kernel)
			biases = biasVariable([512])
			out = tf.nn.bias_add(conv, biases)
			self.conv4_2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		print(self.conv4_2.shape)
		#maxPool4
		self.pool4 = maxPool2x1(self.conv4_2, 'pool4')
		print(self.pool4.shape)
		#conv5 w/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv5') as scope:
			kernel = weightVariable([2, 2, 512, 512])
			conv = conv2d(self.pool4, kernel, 'VALID')
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=True)
			self.conv5= tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		
		print(self.conv5.shape)
		#transpose
		self.transposed = tf.transpose(self.conv5, perm=[2, 0, 3, 1], name='transposed')
		print(self.transposed.shape)
		#reshape
		self.view = tf.reshape(self.transposed, [24, -1, 512], name='view')
		print(self.view.shape)
		#split to get a list of 'n_steps' tensors of shape [n_batches, n_inputs]
		self.splitedtable = tf.split(self.transposed, 24, 0, name='splitedtable')
		print(self.splitedtable[0].shape)
		self.splitedtable = [tf.reshape(x, [-1, 512]) for x in self.splitedtable]
	def lstmLayers(self):
		#biLSTM1
		print(len(self.splitedtable))
		print(self.splitedtable[0].shape)
		with tf.name_scope('biLSTM1') as scope:
			self.biLstm1 = biLSTM(self.splitedtable, 512, 256, scope)
			# weights = tf.Variable(tf.random_normal([256*2, 256]))
			# biases = tf.Variable(tf.random_normal([256]))
			# self.biLstm1 = tf.nn.bias_add(tf.matmul(biLstm, weights), biases)
			# self.parameters += [weights, biases]
		# self.biLstm1 = tf.split(self.biLstm1, 24, 0, name='splitedtable')
		# print(self.biLstm1.shape)
		#biLSTM2
		with tf.name_scope('biLSTM2') as scope:
			self.biLstm2 = biLSTM(self.biLstm1, 256, 256, scope)
			# weights = tf.Variable(tf.random_normal([256*2, 37]))
			# biases = tf.Variable(tf.random_normal([37]))
			# self.biLstm2 = tf.nn.bias_add(tf.matmul(biLstm[-1], weights), biases)
			# self.parameters += [weights, biases]
		#stack table
		# self.joinedtable = tf.stack(self.biLstm2, 0, name='joinedtable')
		# print(self.joinedtable.shape)
		#softmax
		self.softmax = tf.nn.softmax(self.joinedtable, -1)
		print(self.softmax.shape)
	def loadWeights(self, weightFile):
		saver = tf.train.Saver()
		saver.restore(self.sess, weightFile)
		print("Model restored")
	def saveWeights(self, weightFile):
		saver = tf.train.Saver()
		savePath = saver.save(self.sess, weightFile)
		print("Model saved at: %s" % savePath)
		return savePath

class CtcCriterion:
	def __init__(self, result, target, nSamples):
		self.result = result
		self.target = target
		self.nSamples = nSamples
		self.createCtcCriterion()
		self.decodeCtc()
	def createCtcCriterion(self):
		self.loss = tf.nn.ctc_loss(self.target, self.result, self.nSamples)
		self.cost = tf.reduce_mean(self.loss)
	def decodeCtc(self):
		self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.result, self.nSamples)
		self.learningRate = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.target))

