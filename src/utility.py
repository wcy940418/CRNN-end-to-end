from __future__ import print_function
import os

def labelInt2Char(n):
	if n >= 0 and n <=9:
		c = chr(n + 48)
	elif n >= 10 and n<= 35:
		c = chr(n  + 97 - 10)
	elif n == 36:
		c = ''
	return c

def labelInt2CharWithBlank(n):
	if n >= 0 and n <=9:
		c = chr(n + 48)
	elif n >= 10 and n<= 35:
		c = chr(n  + 97 - 10)
	elif n == 36:
		c = '-'
	return c

def simpleDecoderWithBlank(results):
	labels = []
	for i in range(len(results)):
		label = ''
		for j in range(len(results[i])):
			label += labelInt2CharWithBlank(results[i][j])
		labels.append(label)
	return labels

def simpleDecoder(p):
	labels = simpleDecoderWithBlank(p)
	results = []
	for label in labels:
		temp_s = ''
		for i in range(len(label)):
			if label[i] != '-' and not(i > 0 and label[i] == label[i-1]):
				temp_s += label[i]
		results.append(temp_s)
	return results

def convertSparseArrayToStrs(p):
	# print(p[0].shape, p[1].shape, p[2].shape)
	# print(p[2][0], p[2][1])
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

def eval_accuracy(pred, true):
	n = len(pred)
	equal = 0
	for i in range(n):
		if pred[i] == true[i]:
			equal += 1
	return float(equal) / n

def checkPointLoader(modelDir):
	#ckpts folders format: 'ckpt-(steps(8digits))'
	dirs = [x for x in os.listdir(modelDir) if os.path.isdir(os.path.join(modelDir, x)) and 'ckpt' in x]
	if len(dirs) == 0:
		return None
	dirs = sorted(dirs, key = lambda x:int(x.split('-')[-1]), reverse=True)
	if len(dirs) == 1:
		print("There is a check point. Please type in ENTER to load it or \"new\" to start a new training:")
	else:
		print("There are %d check points. Please type in the # of check points or ENTER to load first check point or \"new\" to start a new training:" % len(dirs))
	for i in range(len(dirs)):
		print("%02d. %s" % (i, dirs[i]))
	print(">>", end="")
	num = raw_input()
	if num == '':
		return os.path.join(modelDir, dirs[0], dirs[0])
	elif num =='new':
		return None
	else:
		try:
			num = int(num)
			return os.path.join(modelDir, dirs[int(num)], dirs[int(num)])
		except ValueError:
			print("Please type in a valid number")

if __name__ == '__main__':
	print(checkPointLoader(os.path.abspath(os.path.join('..', 'model', 'ckpt'))))