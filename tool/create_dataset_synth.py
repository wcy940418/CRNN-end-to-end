import os, sys
import lmdb
import numpy as np
import base64

def writeCache(env, cache):
	with env.begin(write=True) as txn:
		for k, v in cache.iteritems():
			txn.put(k, v)


def createDataset(outputPath, configFile):
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
	"python create_dataset.py <config text file path>" '''
	configFile = sys.argv[1]
	outputPath = configFile.split('.')[0] + '_data'
	createDataset(outputPath, configFile)
