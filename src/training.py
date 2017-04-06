import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import CRNN, CtcCriterion
from dataset import DatasetLmdb, SynthLmdb
import tensorflow as tf
import numpy as np
import signal
import utility
import sys
import time

class Conf:
	def __init__(self):
		self.nClasses = 36
		self.trainBatchSize = 64
		self.evalBatchSize = 200
		self.testBatchSize = 10
		self.maxIteration = 2000000
		self.displayInterval = 1
		self.evalInterval = 10
		self.testInterval = 20
		self.saveInterval = 50000
		self.modelDir = os.path.abspath(os.path.join('..', 'model', 'ckpt'))
		# self.dataSet = os.path.join('..', 'data', 'Synth')
		# self.auxDataSet = os.path.join('..', 'data', 'aux_Synth')
		self.dataSet = os.path.join('..', 'data', 'IIIT5K')
		self.maxLength = 24


if __name__ == '__main__':
	gConfig = Conf()
	sess = tf.InteractiveSession()

	ckpt = utility.checkPointLoader(gConfig.modelDir)
	imgs = tf.placeholder(tf.float32, [None, 32, 100])
	decode_labels = tf.sparse_placeholder(tf.int32)
	labels = tf.placeholder(tf.int32,[None])
	target_seq_lengths = tf.placeholder(tf.int32, [None])
	input_seq_lengths = tf.placeholder(tf.int32, [None])
	isTraining = tf.placeholder(tf.bool)
	keepProb = tf.placeholder(tf.float32)
	pred_labels = tf.placeholder(tf.string, [None])
	true_labels = tf.placeholder(tf.string, [None])

	trainSeqLength = [gConfig.maxLength for i in range(gConfig.trainBatchSize)]
	testSeqLength = [gConfig.maxLength for i in range(gConfig.testBatchSize)]
	evalSeqLength = [gConfig.maxLength for i in range(gConfig.evalBatchSize)]

	crnn = CRNN(imgs, gConfig, isTraining, keepProb, sess)
	ctc = CtcCriterion(crnn.prob, input_seq_lengths, labels, target_seq_lengths, pred_labels, true_labels)
	global_step = tf.Variable(0)
	optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(ctc.cost, global_step=global_step)
	if ckpt is None:
		init = tf.global_variables_initializer()
		sess.run(init)
		step = 0
	else:
		crnn.loadModel(ckpt)
		step = sess.run([global_step])

	data = DatasetLmdb(gConfig.dataSet)
	# data = SynthLmdb(gConfig.dataSet, gConfig.auxDataSet)
	
	trainAccuracy = 0

	def signal_handler(signal, frame):
		print('You pressed Ctrl+C!')
		crnn.saveModel(gConfig.modelDir, step)
		print("%d steps trained model has saved" % step)
		sys.exit(0)
	signal.signal(signal.SIGINT, signal_handler)
	start_time = time.time()
	t = start_time
	while True:
		#train
		batchSet, labelSet, seqLengths = data.nextBatch(gConfig.trainBatchSize)
		cost, _, step = sess.run([ctc.cost, optimizer, global_step],feed_dict={
					crnn.inputImgs:batchSet, 
					crnn.isTraining:True,
					crnn.keepProb:1.0,
					ctc.lossTarget:labelSet[1], 
					ctc.targetSeqLengths:seqLengths,
					ctc.inputSeqLengths:trainSeqLength
					})
		if step % gConfig.displayInterval == 0:
			time_elapse = time.time() - t
			t = time.time()
			total_time = time.time() - start_time
			print("step: %s, cost: %s, step time: %.2fs, total time: %.2fs" % (step, cost, time_elapse, total_time))
		#eval accuarcy
		if step != 0 and step % gConfig.evalInterval == 0:
			batchSet, labelSet, seqLengths = data.nextBatch(gConfig.evalBatchSize)
			# print(batchSet.shape, labelSet.shape)

			p = sess.run(crnn.rawPred, feed_dict={
								crnn.inputImgs:batchSet, 
								crnn.isTraining:False,
								crnn.keepProb:1.0,
								ctc.inputSeqLengths:evalSeqLength
								})
			original = utility.convertSparseArrayToStrs(labelSet)
			predicted = utility.simpleDecoder(p)
			# trainAccuracy = sess.run([ctc.accuracy], feed_dict={
			# 						ctc.pred_labels: predicted,
			# 						ctc.true_labels: original
			# 						})
			trainAccuracy = utility.eval_accuracy(predicted, original)
			print("step: %d, training accuracy %f" % (step, trainAccuracy))
		#small test
		if step != 0 and step % gConfig.testInterval == 0:
			batchSet, labelSet, seqLengths = data.nextBatch(gConfig.testBatchSize)
			p = sess.run(crnn.rawPred, feed_dict={
								crnn.inputImgs:batchSet, 
								crnn.isTraining:False,
								crnn.keepProb:1.0,
								ctc.inputSeqLengths:testSeqLength
								})
			original = utility.convertSparseArrayToStrs(labelSet)
			predictedWithBlank = utility.simpleDecoderWithBlank(p)
			predicted = utility.simpleDecoder(p)
			for i in range(len(original)):
				print("original: %s, predicted(no decode): %s, predicted: %s" % (original[i], predictedWithBlank[i], predicted[i]))
		if step >= gConfig.maxIteration:
			print("%d training has completed" % gConfig.maxIteration)
			crnn.saveModel(gConfig.modelDir, step)
			sys.exit(0)
		if step != 0 and step % gConfig.saveInterval == 0:
			print("%d training has saved" % step)
			crnn.saveModel(gConfig.modelDir, step)