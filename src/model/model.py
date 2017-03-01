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
	def __init__(self, inputImgs, conf, weights=None, session=None):
		self.inputImgs = inputImgs
		self.saver = tf.train.Saver()
		self.sess = session
		self.convLayers()
		self.lstmLayers()
		self.prob = self.softmax
		self.config = conf
		self.isTraining = True
		if sess is not None and weights is not None:
			loadWeights(weights)
	def convLayers(self):
		#preprocess
		with tf.name_scope('preprocess') as scope:
			images = tf.reshape(self.inputImgs, [-1, 32, 100, 1])
			#addConst = tf.constant(-128.0, shape=images.shape.as_list())
			#mulConst = tf.constant(1.0 / 128, shape=images.shape.as_list())
			#images = tf.add(images, addConst)
			#self.imgs = tf.multiply(images, mulConst, name=scope)
			images = np.add(images, -128.0)
			images = np.multiply(images, 1.0/128.0)
			self.imgs = images
		#conv1
		with tf.name_scope('conv1') as scope:
			kernel = weightVariable([3, 3, 1, 64])
			conv = conv2d(self.imgs, kernel)
			biases = biasVariable([64])
			out = tf.nn.bias_add(conv, biases)
			self.conv1 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool1
		self.pool1 = maxPool2x2(self.conv1, 'pool1')
		#conv2
		with tf.name_scope('conv2') as scope:
			kernel = weightVariable([3, 3, 64, 128])
			conv = conv2d(self.pool1, kernel)
			biases = biasVariable([128])
			out = tf.nn.bias_add(conv, biases)
			self.conv2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool2
		self.pool2 = maxPool2x2(self.conv2, 'pool2')
		#conv3_1 w/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv3_1') as scope:
			kernel = weightVariable([3, 3, 128, 256])
			conv = conv2d(self.pool2, kernel)
			biases = biasVariable([256])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=self.isTraining)
			self.conv3_1 = tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		#conv3_2
		with tf.name_scope('conv3_2') as scope:
			kernel = weightVariable([3, 3, 256, 256])
			conv = conv2d(self.conv3_1, kernel)
			biases = biasVariable([256])
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool3
		self.pool3 = maxPool2x1(self.conv3_2, 'pool3')
		#conv4_1 w/batch_norm
		with tf.name_scope('conv4_1') as scope:
			kernel = weightVariable([3, 3, 256, 512])
			conv = conv2d(self.pool3, kernel)
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=self.isTraining)
			self.conv4_1 = tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		#conv4_2 wo/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv4_2') as scope:
			kernel = weightVariable([3, 3, 512, 512])
			conv = conv2d(self.conv4_1, kernel)
			biases = biasVariable([512])
			out = tf.nn.bias_add(conv, biases)
			self.conv4_2 = tf.nn.relu(out, name=scope)
			#self.parameters += [kernel, biases]
		#maxPool4
		self.pool4 = maxPool2x1(self.conv4_2, 'pool4')
		#conv5 w/batch_norm(This part is same as source code, not paper)
		with tf.name_scope('conv5') as scope:
			kernel = weightVariable([2, 2, 512, 512])
			conv = conv2d(self.pool4, kernel, 'VALID')
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=self.isTraining)
			self.conv5= tf.nn.relu(batch_norm_out, name=scope)
			#self.parameters += [kernel, biases]
		#reshape
		self.view = tf.reshape(self.conv5, [self.conv5.shape[0], 512, 26], name='view')
		#transpose
		self.transposed = tf.transpose(self.conv5, perm=[2, 0, 1], name='transposed')
		#split to get a list of 'n_steps' tensors of shape [n_batches, n_inputs]
		self.splitedtable = tf.split(self.transposed, 26, 0, name='splitedtable')
	def lstmLayers(self):
		#biLSTM1
		with tf.name_scope('biLSTM1') as scope:
			biLstm = biLSTM(self.splitedtable, 512, 256, scope)
			weights = tf.Variable(tf.random_normal([2*512, 256]))
			biases = tf.Variable(tf.random_normal([256]))
			self.biLstm1 = tf.bias_add(tf.matmul(biLstm, weights), biases)
			# self.parameters += [weights, biases]
		#biLSTM2
		with tf.name_scope('biLSTM2') as scope:
			biLstm = biLSTM(self.biLstm1, 256, 256, scope)
			weights = tf.Variable(tf.random_normal([2*256, self.config.nClasses + 1]))
			biases = tf.Variable(tf.random_normal([self.config.nClasses + 1]))
			self.biLstm2 = tf.bias_add(tf.matmul(biLstm, weights), biases)
			# self.parameters += [weights, biases]
		#stack table
		self.joinedtable = tf.stack(self.biLstm2, 0, name='joinedtable')
		#softmax
		self.softmax = tf.nn.softmax(self.joinedtable, -1)
	def loadWeights(self, weightFile):
		self.saver.restore(self.sess, weightFile)
		print("Model restored")
	def saveWeights(self, weightFile):
		savePath = self.saver.save(self.sess, weightFile)
		print("Model saved at: %s" % savePath)
		return savePath
	def setTraining(self):
		self.isTraining = True
	def setEval(self):
		self.isTraining = False

class CtcCriterion:
	def __init__(self, result, target, nSamples):
		self.input = result
		self.target = target
		self.nSamples = nSamples
		self.createCtcCriterion()
		self.decodeCtc()
	def createCtcCriterion(self):
		self.loss = tf.nn.ctc_loss(self.input, self.target, self.nSamples)
		self.cost = tf.reduce_mean(self.loss)
	def decodeCtc(self):
		self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.input, self.nSamples)
		self.learningRate = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0]. tf.int32), self.target))

