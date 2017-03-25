import lmdb
import os
import tensorflow as tf
import random
import numpy as np
import cv2


class DatasetLmdb:
	def __init__(self, lmdbPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))

	def ascii2Label(self, ascii):
		if ascii >= 48 and ascii <=57:
			c = ascii - 48
		elif ascii >= 65 and ascii <=90:
			c = ascii - 65 +10
		elif ascii >=97 and ascii <=122:
			c = ascii - 97 +10
		return c

	def str2intLable(self, strs, maxLength):
		assert type(strs) is list
		nums = len(strs)
		indices = []
		values = []
		dense_shape = [nums, maxLength]
		for i in range(nums):
			for j in range(maxLength):
				indices.append([i, j])
				if j < len(strs[i]):
					values.append(self.ascii2Label(ord(strs[i][j])))
				else:
					values.append(36)
		indices = np.asarray(indices, dtype=np.int32)
		values = np.asarray(values, dtype=np.int32)
		dense_shape = np.asarray(dense_shape, dtype=np.int32)
		return indices, values, dense_shape

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
		labels = self.str2intLable(labelList, 24)
		return (images, labels)

class SynthLmdb:
	def __init__(self, lmdbPath,dataPath):
		self.env = lmdb.open(lmdbPath, map_size=1099511627776)
		with self.env.begin() as txn:
			self.nSamples = int(txn.get('num-samples'))
		self.dataPath = dataPath

	def ascii2Label(self, ascii):
		if ascii >= 48 and ascii <=57:
			c = ascii - 48
		elif ascii >= 65 and ascii <=90:
			c = ascii - 65 +10
		elif ascii >=97 and ascii <=122:
			c = ascii - 97 +10
		return c

	def str2intLable(self, strs, maxLength):
		assert type(strs) is list
		nums = len(strs)
		indices = []
		values = []
		dense_shape = [nums, maxLength]
		for i in range(nums):
			for j in range(maxLength):
				indices.append([i, j])
				if j < len(strs[i]):
					values.append(self.ascii2Label(ord(strs[i][j])))
				else:
					values.append(36)
		indices = np.asarray(indices, dtype=np.int32)
		values = np.asarray(values, dtype=np.int32)
		dense_shape = np.asarray(dense_shape, dtype=np.int32)
		return indices, values, dense_shape

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
		labels = self.str2intLable(labelList, 24)
		return (images, labels)

if __name__ == '__main__':
	db  = SynthLmdb("../data/Synth/test_data", "../data/Synth")
	batches, labels = db.nextBatch(10)
	print  batches.shape, labels