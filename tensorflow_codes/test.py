from configobj import ConfigObj
from network import conv2d
import tensorflow as tf
from tqdm import tqdm
import glob, os, sys
import network, ops
import data_loader
import numpy as np
import scipy.io
import PIL
import random

def leakyRelu(x, alpha=0.1):
    xx = tf.layers.batch_normalization(x)
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def down(inp, in_ch, out_ch, name):
    mid_ch = (in_ch + out_ch) // 2
    conv1 = leakyRelu(conv2d(inp, [3,3,in_ch,mid_ch], name + '/conv1'))
    conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
    conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
    tmp = tf.pad(inp, [[0,0], [0,0], [0,0], [0,out_ch-in_ch]], 'CONSTANT')
    dic[name] = conv3 + tmp
    return tf.nn.avg_pool(dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

def up(inp, in_ch, out_ch, size, name):
    image = tf.image.resize_bilinear(inp, [size, size])
    image = tf.concat([image, dic[name.replace('up', 'down')]], axis=3)
    mid_ch = (in_ch + out_ch) // 2
    conv1 = leakyRelu(conv2d(image, [3,3,in_ch,mid_ch], name + '/conv1'))
    conv2 = leakyRelu(conv2d(conv1, [3,3,mid_ch,mid_ch], name + '/conv2'))
    conv3 = leakyRelu(conv2d(conv2, [3,3,mid_ch,out_ch], name + '/conv3'))
    return conv3

def build_tower(inp):
    down1 = down(inp,   3,   32, 'Gen_down0')
    down2 = down(down1, 32,  64, 'Gen_down1')
    down3 = down(down2, 64,  128, 'Gen_down2')
    down4 = down(down3, 128,  256, 'Gen_down3')
    ctr = leakyRelu(conv2d(down4, [3,3,256,256], 'Gen_center'))
    size = 512
    up4 = up(ctr, 256*2, 128,  size//8, 'Gen_up3')
    up3 = up(up4, 128*2, 64,  size//4, 'Gen_up2')
    up2 = up(up3, 64*2, 32,  size//2, 'Gen_up1')
    up1 = up(up2, 32*2, 16,  size, 'Gen_up0')
    return conv2d(up1, [3,3,16,3], 'Gen_last_layer')





if __name__ == '__main__':

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        #Specify location of the data to be loaded as .npy files.
        images = glob.glob('test_data_folder/*.npy')
        print(images)
        input_ = tf.placeholder(tf.float32, shape=[512, 512, 3])
        devices = ops.get_available_gpus()
        dic = {}

        with tf.variable_scope('Generator'), tf.device('/gpu:0'):
            tf_output = build_tower((tf.expand_dims(input_, axis=0)))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            #Specify the lodation of the model to be used.
            tf.train.Saver().restore(sess, 'Models/MODEL PATH')
            means = []
            for i in tqdm(range(len(images))):
                #Specify location where the data should be saved.
                path = 'outputs/{}'.format(images[i].split('\\')[-1].split('.')[0])
                image = np.load(images[i])
                image = np.array(image, dtype = np.float32)
                image_temp = image.copy()
                image[:,:,0] = 0.299*image_temp[:,:,0]+0.587*image_temp[:,:,1]+0.114*image_temp[:,:,2]
                image[:,:,1] = (image_temp[:,:,0] - image[:,:,0])*0.713 + 128
                image[:,:,2] = (image_temp[:,:,2] - image[:,:,0])*0.564 + 128

                xx=image[:,:,:]

                z = sess.run(tf_output, feed_dict={input_: xx})
                z = np.squeeze(z)
                z_temp = z.copy()
                z[:,:,0] = z_temp[:,:,0] + 1.403* (z_temp[:,:,1] - 128)
                z[:,:,1] = z_temp[:,:,0] - 0.714* (z_temp[:,:,1] - 128) - 0.344*(z_temp[:,:,2] - 128)
                z[:,:,2] = z_temp[:,:,0] + 1.773* (z_temp[:,:,2] - 128)

                scipy.io.savemat(path, {'input':image_temp , 'output': z})

