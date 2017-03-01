import model
import dataset

class Conf:
	self.nClasses = 36
	self.trainBatchSize = 100
	self.testBatchSize = 200
	self.maxIteration = 20000
	self.displayInterval = 200
	self.testInteval = 100


gConfig = Conf()
sess = tf.Session()

imgs = tf.placeholder(tf.float32, [None, 32, 100, 1])
labels = tf.sparse_placeholder(tf.int32)
batches = tf.placeholder(tf.int32, [None])

crnn = CRNN(imgs, gConfig, None, sess)
ctc = CtcCriterion(crnn.prob, labels, batches)
optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(ctc.cost)
data = DatasetLmdb('../data')

init = tf.initialize_all_variables()
sess.run(init)

if __name__ == '__main__':
	trainAccuracy = 0
	for i in range(gConfig.maxIteration):
		if i % gConfig.testInteval == 0:
			batchSet, labelSet = data.nextBatch(gConfig.testBatchSize)
			trainAccuracy = ctc.learningRate.eval(feed_dict={crnn.imgs:batchSet, 
									ctc.target:labelSet, 
									ctc.nSamples:[gConfig.testBatchSize]})
			print("step %d, training accuarcy %g" % (i, trainAccuracy))
		batchSet, labelSet = data.nextBatch(gConfig.trainBatchSize)
		optimizer.run(feed_dict={crnn.imgs:batchSet, 
					ctc.target:labelSet, 
					ctc.nSamples:[gConfig.testBatchSize]})