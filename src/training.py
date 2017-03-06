from model import CRNN, CtcCriterion
from dataset import DatasetLmdb
import os
import tensorflow as tf
import numpy as np

class Conf:
	def __init__(self):
		self.nClasses = 36
		self.trainBatchSize = 100
		self.testBatchSize = 200
		self.maxIteration = 3000
		self.displayInterval = 200
		self.testInteval = 100
		self.modelParFile = './crnn.model'
		self.dataSet = '../data'
		self.maxLength = 24

def labelInt2Char(n):
	if n >= 0 and n <=9:
		c = chr(n + 48)
	elif n >= 10 and n<= 35:
		c = chr(n  + 97 - 10)
	elif n == 36:
		c = ''
	return c

def convertSparseArrayToStrs(p):
	print(p[0].shape, p[1].shape, p[2].shape)
	print(p[2][0], p[2][1])
	results = []
	labels = []
	for i in range(p[2][0]):
		results.append([36 for x in range(p[2][1])])
	for i in range(p[0].shape[0]):
		x, y = p[0][i]
		results[x][y] = p[1][i]
	for i in range(len(results)):
		label = ''
		for j in range(len(results[i])):
			label += labelInt2Char(results[i][j])
		labels.append(label)
	return labels

if __name__ == '__main__':
	gConfig = Conf()
	sess = tf.InteractiveSession()

	weights = None
	if os.path.isfile(gConfig.modelParFile):
		weights = gConfig.modelParFile
	imgs = tf.placeholder(tf.float32, [None, 32, 100])
	labels = tf.sparse_placeholder(tf.int32)
	batches = tf.placeholder(tf.int32, [None])
	isTraining = tf.placeholder(tf.bool)

	trainSeqLength = [gConfig.maxLength for i in range(gConfig.trainBatchSize)]
	testSeqLength = [gConfig.maxLength for i in range(gConfig.testBatchSize)]
	evalSeqLength = [gConfig.maxLength for i in range(10)]

	crnn = CRNN(imgs, gConfig, isTraining, weights, sess)
	ctc = CtcCriterion(crnn.prob, labels, batches)
	optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(ctc.cost)
	data = DatasetLmdb(gConfig.dataSet)

	init = tf.global_variables_initializer()
	sess.run(init)
	trainAccuracy = 0
	for i in range(gConfig.maxIteration + 1):
		if i != 0 and i % gConfig.testInteval == 0:
			batchSet, labelSet = data.nextBatch(gConfig.testBatchSize)
			# print(batchSet.shape, labelSet.shape)
			trainAccuracy = ctc.learningRate.eval(feed_dict={
									crnn.inputImgs:batchSet, 
									crnn.isTraining:False,
									ctc.target:labelSet, 
									ctc.nSamples:testSeqLength
									})
			print("step %d, training accuarcy %g" % (i, trainAccuracy))
		batchSet, labelSet = data.nextBatch(gConfig.trainBatchSize)
		cost, _ = sess.run([ctc.cost, optimizer],feed_dict={
					crnn.inputImgs:batchSet, 
					crnn.isTraining:True,
					ctc.target:labelSet, 
					ctc.nSamples:trainSeqLength
					})
		print("step: %s, cost: %s" % (i, cost))
		if i != 0 and i % 500 == 0:
			batchSet, labelSet = data.nextBatch(10)
			p = sess.run(ctc.decoded, feed_dict={
								crnn.inputImgs:batchSet, 
								crnn.isTraining:False,
								ctc.target:labelSet, 
								ctc.nSamples:evalSeqLength
								})
			original = convertSparseArrayToStrs(labelSet)
			predicted = convertSparseArrayToStrs(p[0])
			for i in range(len(original)):
				print("original: %s, predicted: %s" % (original[i], predicted[i]))
	crnn.saveWeights(gConfig.modelParFile)