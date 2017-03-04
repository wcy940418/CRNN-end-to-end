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
		self.maxIteration = 1000
		self.displayInterval = 200
		self.testInteval = 100
		self.modelParFile = './model/crnn.model'
		self.dataSet = '../data'
		self.maxLength = 24

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

	crnn = CRNN(imgs, gConfig, isTraining, weights, sess)
	ctc = CtcCriterion(crnn.prob, labels, batches)
	optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(ctc.cost)
	data = DatasetLmdb(gConfig.dataSet)

	init = tf.global_variables_initializer()
	sess.run(init)
	trainAccuracy = 0
	for i in range(gConfig.maxIteration):
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
	crnn.saveWeights(self.modelParFile)