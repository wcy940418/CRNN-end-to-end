import tensorflow as tf

def weightVariable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
	return tf.Variable(initial)

def biasVariable(shape):
	initial = tf.constant(0.1, shape=shape, name='biases')
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=pool_name)

def maxPool1x2(x, pool_name=None):
	return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=pool_name)


class CRNN:
	def __init__(self, imgs, weights=None, sess-None):
		self.imgs = imgs
		self.convLayers()
		self.prob = None
		if weights is not None and sess is not None:
			self.isTraining = True
	def convLayers(self):
		#conv1
		with tf.name_scope('conv1') as scope:
			kernel = weightVariable([3, 3, 1, 64])
			conv = conv2d(image, kernel)
			biases = biasVariable([64])
			out = tf.nn.bias_add(conv, biases)
			self.conv1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		#maxPool1
		self.pool1 = maxPool2x2(self.conv1, 'pool1')
		#conv2
		with tf.name_scope('conv2') as scope:
			kernel = weightVariable([3, 3, 64, 128])
			conv = conv2d(self.pool1, kernel)
			biases = biasVariable([128])
			out = tf.nn.bias_add(conv, biases)
			self.conv2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		#maxPool2
		self.pool2 = maxPool2x2(self.conv2, 'pool2')
		#conv3_1
		with tf.name_scope('conv3_1') as scope:
			kernel = weightVariable([3, 3, 128, 256])
			conv = conv2d(self.pool2, kernel)
			biases = biasVariable([256])
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		#conv3_2
		with tf.name_scope('conv3_2') as scope:
			kernel = weightVariable([3, 3, 256, 256])
			conv = conv2d(self.conv3_1, kernel)
			biases = biasVariable([256])
			out = tf.nn.bias_add(conv, biases)
			self.conv3_2 = tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		#maxPool3
		self.pool3 = maxPool1x2(self.conv3_2, 'pool3')
		#conv4_1 w/batch_norm
		with tf.name_scope('conv4_1') as scope:
			kernel = weightVariable([3, 3, 256, 512])
			conv = conv2d(self.pool3, kernel)
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=self.isTraining)
			self.conv4_1 = tf.nn.relu(batch_norm_out, name=scope)
			self.parameters += [kernel, biases]
		#conv4_2 w/batch_norm
		with tf.name_scope('conv4_2') as scope:
			kernel = weightVariable([3, 3, 512, 512])
			conv = conv2d(self.conv4_1, kernel)
			biases = biasVariable([512])
			conv_out = tf.nn.bias_add(conv, biases)
			batch_norm_out = tf.contrib.layers.batch_norm(conv_out, center=False, is_training=self.isTraining)
			self.conv4_2 = tf.nn.relu(batch_norm_out, name=scope)
			self.parameters += [kernel, biases]
		#maxPool4
		self.pool4 = maxPool1x2(self.conv4_2, 'pool4')
		#conv5
		with tf.name_scope('conv5') as scope:
			kernel = weightVariable([3, 3, 512, 512])
			conv = conv2d(self.pool4, kernel)
			biases = biasVariable([512])
			out = tf.nn.bias_add(conv, biases)
			self.conv5= tf.nn.relu(out, name=scope)
			self.parameters += [kernel, biases]
		
