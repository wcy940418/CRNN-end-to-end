import os, sys
import lmdb
import numpy as np
import base64

def checkImageIsValid(imagePath):
	if imageBin is None:
		return False
	img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	imgH, imgW = img.shape[0], img.shape[1]
	if imgH * imgW == 0:
		return False
	return True

def writeCache(env, cache):
	with env.begin(write=True) as txn:
		for k, v in cache.iteritems():
			txn.put(k, v)


def createDataset(outputPath, configFile, imgDir):
	"""
	Create LMDB dataset for CRNN training.

	ARGS:
		outputPath    : LMDB output path
		imagePathList : list of image path
		labelList     : list of corresponding groundtruth texts
		lexiconList   : (optional) list of lexicon lists
		checkValid    : if true, check the validity of every image
	"""
	env = lmdb.open(outputPath, map_size=1099511627776)
	cache = {}
	cnt = 1
	with open(configFile, 'r') as f:
		for line in f:
			image_path = line.strip().split(' ')[0]
			img = os.path.join(imgDir, image_path)
			if checkImageIsValid(img) :
				key = "%08d" % cnt
				cache[key] = image_path
				if cnt % 10000 == 0 and cnt != 0:
					writeCache(env, cache)
					cache = {}
					print "Written %d images" % cnt
				cnt += 1
	nSamples = cnt-1
	cache['num-samples'] = str(nSamples)
	writeCache(env, cache)
	print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
	'''ro run create_dataset, use command: 
	"python create_dataset.py <config text file path> <image files dir path>" '''
	configFile = sys.argv[1]
	imgDir = sys.argv[2]
	outputPath = './data'
	createDataset(outputPath, configFile, imgDir)
