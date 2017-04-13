from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import warpctc_tensorflow

def weightVariable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	return tf.Variable(initial)

def biasVariable(shape):
	initial = tf.constant(0.0, shape=shape, name='biases')
	return tf.Variable(initial)

def conv2d(x, W, pad='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=pad)

def maxPool2x2(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=pool_name)

def maxPool2x1(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME', name=pool_name)

def biLSTM(x, nInputs, nHidden, keep_prob, seqLengths):
	# using standard lstm cell without peephole
	lstmFwCell = tf.contrib.rnn.LSTMCell(nHidden, forget_bias=1.0, state_is_tuple=True)
	lstmBwCell = tf.contrib.rnn.LSTMCell(nHidden, forget_bias=1.0, state_is_tuple=True)
	# lstmBwCell = tf.contrib.rnn.DropoutWrapper(lstmBwCell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
	# lstmFwCell = tf.contrib.rnn.DropoutWrapper(lstmFwCell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
	# using static bidirectional rnn
	# outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstmFwCell, lstmBwCell, x, dtype=tf.float32)
	# using dynamic bidirectional rnn
	outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, x, sequence_length=seqLengths, dtype=tf.float32, time_major=True)
	# return outputs
	return tf.concat(outputs, 2)

class CRNN:
	def __init__(self, inputImgs, conf, isTraining, keepProb, rnnSeqLengths, session=None):
		self.inputImgs = inputImgs
		self.sess = session
		self.config = conf
		self.isTraining = isTraining
		self.keepProb = keepProb
		self.rnnSeqLengths = rnnSeqLengths
		self.convLayers()
		self.lstmLayers()
		self.prob = self.biLstm2
	def convLayers(self):
		#preprocess
		with tf.variable_scope('preprocess') as scope:
			images = self.inputImgs
			images = tf.reshape(images, [-1, 32, 100, 1])
			images = np.add(images, -128.0)
			images = np.multiply(images, 1.0/128.0)
			self.imgs = images #[?, 32, 100, 1]
			tf.summary.image('input image', self.imgs, 1)
		#conv1
		with tf.variable_scope('conv1') as scope:
			kernel = weightVariable([3, 3, 1, 64])
			conv = conv2d(self.imgs, kernel)
			biases = biasVariable([64])
			out = tf.nn.bias_add(conv, biases)
			self.conv1 = tf.nn.relu(out)#[?, 32, 100, 64]
			out_sample = self.conv1[:,:,:,:1]
			tf.summary.image('conv1_1st sample', out_sample, 1)
			tf.summary.histogram('conv1_weights', kernel)
		#maxPool1
		self.pool1 = maxPool2x2(self.conv1, 'pool1') # [?, 16, 50, 64]
		#conv2
		with tf.variable_scope('conv2') as scope:
			kernel = weightVariable([3, 3, 64, 128])
			conv = conv2d(self.pool1, kernel)
			biases = biasVariable([128])
			out = tf.nn.bias_add(conv, biases)
			self.conv2 = tf.nn.relu(out)# [? 16, 50, 128]
		#maxPool2
		self.pool2 = maxPool2x2(self.conv2, 'pool2') #[?, 8, 25, 128]
		#conv3_1 w/batch_norm(This part is same as source code, not paper)
		with tf.variable_scope('conv3_1') as scope:
			kernel = weightVariable([3, 3, 128, 256])
			conv = conv2d(self.pool2, kernel)
			biases = biasVariable([256])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.layers.batch_normalization(conv_out, training=self.isTraining)
			self.conv3_1 = tf.nn.relu(batch_norm_out)# [?, 8, 25, 256]
		#conv3_2
		with tf.variable_scope('conv3_2') as scope:
			kernel = weightVariable([3, 3, 256, 256])
			conv = conv2d(self.conv3_1, kernel)
			biases = biasVariable([256])
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out)# [?, 8, 25, 256]
		#maxPool3
		self.pool3 = maxPool2x1(self.conv3_2, 'pool3')# [?, 4, 25, 256]
		print(self.pool3.shape)	
		#conv4_1 w/batch_norm
		with tf.variable_scope('conv4_1') as scope:
			kernel = weightVariable([3, 3, 256, 512])
			conv = conv2d(self.pool3, kernel)
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.layers.batch_normalization(conv_out, training=self.isTraining)
			self.conv4_1 = tf.nn.relu(batch_norm_out)#[?, 4, 25, 512]
		#conv4_2 wo/batch_norm(This part is same as source code, not paper)
		with tf.variable_scope('conv4_2') as scope:
			kernel = weightVariable([3, 3, 512, 512])
			conv = conv2d(self.conv4_1, kernel)
			biases = biasVariable([512])
			out = tf.nn.bias_add(conv, biases)
			self.conv4_2 = tf.nn.relu(out) #[?, 4, 25, 512]
		#maxPool4
		self.pool4 = maxPool2x1(self.conv4_2, 'pool4') #[?, 2, 25, 512]
		print(self.pool4.shape)	
		#conv5 w/batch_norm(This part is same as source code, not paper)
		with tf.variable_scope('conv5') as scope:
			kernel = weightVariable([2, 2, 512, 512])
			conv = conv2d(self.pool4, kernel, 'VALID')
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.layers.batch_normalization(conv_out, training=self.isTraining)
			self.conv5= tf.nn.relu(batch_norm_out) #[?, 1, 24, 512]
		print(self.conv5.shape)	
		''' 
		# work for static bidirectional rnn
		#transpose
		self.transposed = tf.transpose(self.conv5, perm=[2, 0, 3, 1], name='transposed')
		#reshape
		self.view = tf.reshape(self.transposed, [24, -1, 512], name='view')
		print(self.view.shape)	
		#split to get a list of 'n_steps' tensors of shape [n_batches, n_inputs]
		self.splitedtable = tf.split(self.view, 24, 0, name='splitedtable')
		self.splitedtable = [tf.reshape(x, [-1, 512]) for x in self.splitedtable]
		print(self.splitedtable[0].shape)	
		'''
		''' 
		# work for static bidirectional rnn
		#reshape
		self.view1 = tf.squeeze(self.conv5, name='view1')
		print(self.view1.shape)	
		#transpose
		self.transposed = tf.transpose(self.view1, perm=[1, 0, 2], name='transposed')
		#reshape
		self.view2 = tf.reshape(self.transposed, [-1, 512], name='view2')
		#split to get a list of 'n_steps' tensors of shape [n_batches, n_inputs]
		self.splitedtable = tf.split(self.view2, 24, 0, name='splitedtable')
		print(self.splitedtable[0].shape)
		'''
		
		# work for dynamic bidirectional rnn
		#transpose
		transposed = tf.transpose(self.conv5, perm=[2, 0, 3, 1], name='transposed')
		#reshape
		self.view = tf.reshape(transposed, [24, -1, 512], name='view')
		print(self.view.shape)
		
	def lstmLayers(self):
		#biLSTM1
		# with tf.variable_scope('biLSTM1') as scope:
		# 	# work for static bidirectional rnn
		# 	# biLstm = biLSTM(self.splitedtable, 512, 256, self.keepProb)
		# 	# joinedtable = tf.stack(biLstm, 0)
		# 	# joinedtable = tf.reshape(joinedtable, [-1, 512])
		# 	# weights = weightVariable([256*2, 256])
		# 	# biases = biasVariable([256])
		# 	# self.biLstm1 = tf.nn.bias_add(tf.matmul(joinedtable, weights), biases)
		# 	# print(self.biLstm1.shape)
		# 	# self.biLstm1 = tf.split(self.biLstm1, 24, 0)

		# 	# work for dynamic bidirectional rnn
		# 	biLstm = biLSTM(self.view, 512, 256, self.keepProb, self.rnnSeqLengths)
		# 	joinedtable = tf.reshape(biLstm, [-1, 512])
		# 	weights = weightVariable([256*2, 256])
		# 	biases = biasVariable([256])
		# 	self.biLstm1 = tf.nn.bias_add(tf.matmul(joinedtable, weights), biases)
		# 	self.biLstm1 = tf.reshape(self.biLstm1, [24, -1, 256])
		# 	print(self.biLstm1.shape)

		#biLSTM2
		with tf.variable_scope('biLSTM2') as scope:
			# work for static bidirectional rnn
			# biLstm = biLSTM(self.biLstm1, 256, 256, self.keepProb)
			# joinedtable = tf.stack(biLstm, 0)
			# joinedtable = tf.reshape(joinedtable, [-1, 512])
			# weights = weightVariable([256*2, 37])
			# biases = biasVariable([37])
			# self.biLstm2 = tf.nn.bias_add(tf.matmul(joinedtable, weights), biases)
			# self.biLstm2 = tf.reshape(self.biLstm2, [24, -1, 37])

			# work for dynamic bidirectional rnn
			lstmout = biLSTM(self.view, 512, 256, self.keepProb, self.rnnSeqLengths)
			joinedtable = tf.reshape(lstmout, [-1, 512])
			weights = weightVariable([256*2, 37])
			biases = biasVariable([37])
			calout = tf.nn.bias_add(tf.matmul(joinedtable, weights), biases)
			self.biLstm2 = tf.reshape(calout, [24, -1, 37])
		pred = tf.transpose(tf.nn.softmax(self.biLstm2), perm=[1, 0, 2])
		self.rawPred = tf.argmax(pred, 2)
		print(self.rawPred.shape)
	def loadModel(self, modelFile):
		saver = tf.train.Saver()
		saver.restore(self.sess, modelFile)
		print("Model restored")
	def saveModel(self, modelFile, step):
		saver = tf.train.Saver()
		save_path = os.path.join(modelFile, "ckpt-%08d" % step)
		if not os.path.isdir(save_path):
			os.mkdir(save_path)
		savePath = saver.save(self.sess, os.path.join(save_path, "ckpt-%08d" % step))
		print("Model saved at: %s" % savePath)
		return savePath

class CtcCriterion:
	def __init__(self, result, inputSeqLengths, lossTarget, targetSeqLengths, pred_labels, true_labels):
		self.result = result
		self.inputSeqLengths = inputSeqLengths
		self.lossTarget = lossTarget
		self.targetSeqLengths = targetSeqLengths
		self.pred_labels = pred_labels
		self.true_labels = true_labels
		self.createCtcCriterion()
		self.decodeCtc()
	def createCtcCriterion(self):
		# using built-in ctc loss calculator
		# self.loss = tf.nn.ctc_loss(self.target, self.result, self.lossSeqLengths)
		# using baidu's warp ctc loss calculator
		self.loss = warpctc_tensorflow.ctc(self.result, self.lossTarget, self.targetSeqLengths, self.inputSeqLengths, blank_label=36)
		self.cost = tf.reduce_mean(self.loss)
	def decodeCtc(self):
		# currently do not use greedy decoder, using naive decoder instead
		self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.result, self.inputSeqLengths)
		# calculate the accuracy
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels, self.true_labels), tf.float32))

