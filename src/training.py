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

if __name__ == '__main__':
	gConfig = Conf()
	sess = tf.InteractiveSession()

	weights = None
	if os.path.isfile(gConfig.modelParFile):
		weights = self.modelParFile
	imgs = tf.placeholder(tf.float32, [None, 32, 100, 1])
	labels = tf.sparse_placeholder(tf.int32)
	batches = tf.placeholder(tf.int32, [None])
	isT = tf.placeholder(tf.bool)

	crnn = CRNN(imgs, gConfig, isT, weights, sess)
	ctc = CtcCriterion(crnn.prob, labels, batches)
	optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(ctc.cost)
	data = DatasetLmdb(gConfig.dataSet)

	init = tf.global_variables_initializer()
	sess.run(init)
	trainAccuracy = 0
	for i in range(gConfig.maxIteration):
		if i % gConfig.testInteval == 0:
			batchSet, labelSet = data.nextBatch(gConfig.testBatchSize)
			trainAccuracy = ctc.learningRate.eval(feed_dict={crnn.imgs:batchSet, 
									crnn.isTraining:False,
									ctc.target:labelSet, 
									ctc.nSamples:gConfig.testBatchSize})
			print("step %d, training accuarcy %g" % (i, trainAccuracy))
		batchSet, labelSet = data.nextBatch(gConfig.trainBatchSize)
		optimizer.run(feed_dict={crnn.imgs:batchSet, 
					crnn.isTraining:True,
					ctc.target:labelSet, 
					ctc.nSamples:gConfig.trainBatchSize})
	crnn.saveWeights(self.modelParFile)