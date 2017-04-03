import lmdb
import os
import tensorflow as tf
import random
import numpy as np
import cv2

def ascii2Label(ascii):
	if ascii >= 48 and ascii <=57:
		c = ascii - 48
	elif ascii >= 65 and ascii <=90:
		c = ascii - 65 +10
	elif ascii >=97 and ascii <=122:
		c = ascii - 97 +10
	return c

def str2intLable(strs):
	assert type(strs) is list
	nums = len(strs)
	maxLength = 0
	indices = []
	values = []
	seqLengths = []
	for i in range(nums):
		length = len(strs[i])
		if length > maxLength:
			maxLength = length
		for j in range(length):
			indices.append([i, j])
			values.append(ascii2Label(ord(strs[i][j])))
		seqLengths.append(length)
	dense_shape = [nums, maxLength]
	indices = np.asarray(indices, dtype=np.int32)
	values = np.asarray(values, dtype=np.int32)
	dense_shape = np.asarray(dense_shape, dtype=np.int32)
	return (indices, values, dense_shape), seqLengths


class DatasetLmdb:
	def __init__(self, lmdbPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))

	def getNumSamples(self):
		return self.nSamples

	def nextBatch(self, batchSize):
		imgW = 100
		imgH = 32
		randomIndex = random.sample(range(1, self.nSamples), batchSize)
		imageList = []
		labelList = []
		imageKeyList = []
		images = []
		errorCounter = 0
		with self.env.begin() as txn:
			for i in range(batchSize):
				idx = randomIndex[i]
				imageKey = 'image-%09d' % idx
				labelKey = 'label-%09d' % idx
				imageBin = txn.get(imageKey)
				labelBin = txn.get(labelKey)
				imageList.append(imageBin)
				labelList.append(labelBin)
				imageKeyList.append(imageKey)
		for i in range(batchSize):
			imageBin = imageList[i]
			imageBuf = np.fromstring(imageBin, dtype=np.uint8)
			decompressedImg = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
			# print decompressedImg.shape
			resized = cv2.resize(decompressedImg, (imgW, imgH))
			# print resized.shape
			images.append(resized)
		images = np.asarray(images)
		labels, seqLengths = str2intLable(labelList)
		return images, labels, seqLengths

class SynthLmdb:
	def __init__(self, lmdbPath,dataPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))
		self.dataPath = dataPath

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
				imageKey = '%08d' % idx
				imagePath = txn.get(imageKey)
				label = os.path.splitext(os.path.split(imagePath)[-1])[0].split('_')[1]
				imageList.append(imagePath)
				labelList.append(label)
		for i in range(batchSize):
			decompressedImg = cv2.imread(os.path.join(self.dataPath, imageList[i]), cv2.IMREAD_GRAYSCALE)
			# print decompressedImg.shape
			resized = cv2.resize(decompressedImg, (imgW, imgH))
			# print resized.shape
			images.append(resized)
		images = np.asarray(images)
		labels, seqLengths = str2intLable(labelList)
		return images, labels, seqLengths

if __name__ == '__main__':
	# db  = SynthLmdb("../data/Synth/test_data", "../data/Synth")
	db  = DatasetLmdb("../data/IIIT5K")
	batches, labels, seqLengths = db.nextBatch(10)
	import utility
	pred = utility.convertSparseArrayToStrs(labels)
	print  batches.shape, pred, seqLengths, labels[2]
	# for b in batches:
	# 	print b.shape
	# 	cv2.imshow("output", b)
	# 	cv2.waitKey(0)