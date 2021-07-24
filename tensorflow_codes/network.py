import tensorflow as tf
import numpy as np
import ops, sys

def conv2d(inp, shp, name, strides=(1,1,1,1), padding='SAME', trainable=True):
    with tf.device('/cpu:0'):
        filters = tf.get_variable(name + '/filters', shp, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/(shp[0]*shp[1]*shp[3]))), trainable=trainable)
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0), trainable=trainable)
    return tf.nn.bias_add(tf.nn.conv2d(inp, filters, strides=strides, padding=padding), biases)

def leakyRelu(x, alpha=0.1):
    xx = tf.layers.batch_normalization(x)
    return tf.nn.relu(xx) - alpha * tf.nn.relu(-xx)

def leakyRelu_d(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def fc_layer(inp, shp, name):
    with tf.device('/cpu:0'):
        weights = tf.get_variable(name + '/weights', shp, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name + '/biases', [shp[-1]], initializer=tf.constant_initializer(0))
    return tf.nn.bias_add(tf.matmul(inp, weights), biases)

def normal_block(inp, name, is_training):
    ch = inp.get_shape().as_list()[-1]
    conv1 = leakyRelu_d(conv2d(inp, [3,3,ch,ch], name + '/conv1'))
    conv2 = leakyRelu_d(conv2d(conv1, [3,3,ch,ch*2], name + '/conv2', strides=(1,2,2,1)))
    return conv2

def _YCbCr2RGB(image):
    X = image[...,0] + 1.403* (image[...,1] - 128)
    Y = image[...,0] - 0.714* (image[...,1] - 128) - 0.344*(image[...,2] - 128)
    Z = image[...,0] + 1.773* (image[...,2] - 128)
    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)

def _normalize(image):
    X = (image[...,0] - tf.reduce_min(image[...,0]))/( tf.reduce_max(image[...,0])- tf.reduce_min(image[...,0]))*255.0
    Y = (image[...,1] - tf.reduce_min(image[...,1]))/( tf.reduce_max(image[...,1])- tf.reduce_min(image[...,1]))*255.0
    Z = (image[...,2] - tf.reduce_min(image[...,2]))/( tf.reduce_max(image[...,2])- tf.reduce_min(image[...,2]))*255.0
    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)

def sobelFilter(inp):
    inp = tf.greater(inp, 0.05)
    inp = tf.cast(inp, tf.float32)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x = tf.nn.conv2d(inp, sobel_x_filter,
                              strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(inp, sobel_y_filter,
                              strides=[1, 1, 1, 1], padding='SAME')
    return tf.cast(tf.cast(tf.abs(filtered_x) + tf.abs(filtered_y), tf.bool), tf.float32)

class Generator(object):

    def __init__(self, inp, config):
        self.dic = {}
        self.config = config
        cur = inp
        print(cur.get_shape())
        for i in range(self.config.n_levels):
            cur = self.down(cur, i)
        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu(conv2d(cur, [3,3,ch,ch], 'Gen_center'))
        for i in range(self.config.n_levels):
            cur = self.up(cur, self.config.n_levels - i - 1)

        self.output = conv2d(cur, [3,3,self.config.n_channels//2,3], 'Gen_last_layer')

    def down(self, inp, lvl):
        name = 'Gen_down{}'.format(lvl)
        in_ch = inp.get_shape().as_list()[-1]
        out_ch = self.config.n_channels if lvl == 0 else in_ch * 2
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
        self.dic[name] = conv3 + tmp
        return tf.nn.avg_pool(self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up(self, inp, lvl):
        name = 'Gen_up{}'.format(lvl)
        size = self.config.image_size >> lvl
        image = tf.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        in_ch = image.get_shape().as_list()[-1]
        out_ch = in_ch // 4
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
        return conv3

class Discriminator(object):
    def __init__(self, inp, config):
        cur = leakyRelu_d(conv2d(inp, [3,3,3,config.n_channels], 'conv1'))
        for i in range(config.n_blocks):
            cur = normal_block(cur, 'n_block{}'.format(i), config.is_training)
        cur = tf.reduce_mean(cur, axis=(1,2))
        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu_d(fc_layer(cur, [ch, ch], 'fcl1'))
        self.output = tf.nn.sigmoid(fc_layer(cur, [ch, 1], 'fcl2'))
