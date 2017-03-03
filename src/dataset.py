import lmdb
import os
import tensorflow as tf
import random
import numpy as np


class DatasetLmdb:
	def __init__(self, lmdbPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))

	def ascii2Label(self, ascii):
		if ascii >= 48 and ascii <=57:
			c = ascii - 47
		elif ascii >= 65 and ascii <=90:
			c = ascii - 64 +10
		elif ascii >=97 and ascii <=122:
			c = ascii - 96 +10
		return c

	def str2intLable(self, strs, maxLength):
		assert type(strs) is list
		nums = len(strs)
		labels = []
		for i in range(nums):
			labels.append([self.ascii2Label(ord(c)) for c in strs[i]])
		labels = np.asarray(labels)
		return labels

	def getNumSamples(self):
		return self.nSamples

	def nextBatch(self, batchSize):
		imgW = 100
		imgH = 32
		randomIndex = random.sample(range(1, self.nSamples), batchSize)
		imageList = []
		labelList = []
		images = []
		with self.env.begin() as txn:
			for i in range(batchSize):
				idx = randomIndex[i]
				imageKey = 'image-%09d' % idx
				labelKey = 'label-%09d' % idx
				imageBin = str(txn.get(imageKey))
				labelBin = str(txn.get(labelKey))
				imageList.append(imageBin)
				labelList.append(labelBin)
		# images = np.ndarray(shape=(batchSize, imgH, imgW, 1), dtype=int)
		for i in range(batchSize):
			imgBin = imageList[i]
			decompressedImg = tf.image.decode_jpeg(imgBin, channels=1)
			images.append(tf.image.resize_images(decompressedImg, [32, 100]))
		images = tf.stack(images, 0)
		images = images.eval()
		labels = self.str2intLable(labelList, 24)
		return (images, labels)



