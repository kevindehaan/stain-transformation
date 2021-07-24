from configobj import ConfigObj
from time import time, sleep
import data_loader
import network
import tensorflow as tf
from tqdm import tqdm
import glob, ops, sys
import numpy as np
import random
from network_1stain import _YCbCr2RGB, _normalize, sobelFilter

def rgb2ycbcr(image):
    image_temp = image
    X = 0.299*image_temp[:,:,:,0]+0.587*image_temp[:,:,:,1]+0.114*image_temp[:,:,:,2]
    Y = (image_temp[:,:,:,0] - X)*0.713 + 128
    Z = (image_temp[:,:,:,2] - X)*0.564 + 128

    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)


def ycbcr2rgb(image):
    X = image[...,0] + 1.403* (image[...,1] - 128)
    Y = image[...,0] - 0.714* (image[...,1] - 128) - 0.344*(image[...,2] - 128)
    Z = image[...,0] + 1.773* (image[...,2] - 128)
    X = tf.expand_dims(X, axis=3)
    Y = tf.expand_dims(Y, axis=3)
    Z = tf.expand_dims(Z, axis=3)
    return tf.concat([X, Y, Z], axis=3)

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.is_training, vc.is_training = True, False
    tc.batch_size, vc.batch_size = 12, 12
    tc.n_channels, vc.n_channels = 16, 16
    tc.image_size, vc.image_size = 256, 256
    tc.n_threads, vc.n_threads = 2, 1
    tc.n_blocks, vc.n_blocks = 5, 5
    tc.n_levels, vc.n_levels = 4, 4
    tc.checkpoint = 1000
    tc.q_limit = 1000
    tc.lamda = 2000.0
    return tc, vc

if __name__ == '__main__':
    #Choose the location of the training and validation images
    train_images = glob.glob('image_dataset/training/label/*.tif')
    valid_images = glob.glob('image_dataset/validation/label/*.tif')

    print(train_images)
    random.shuffle(train_images)

    train_config, valid_config = init_parameters()
    patch_size = train_config.image_size
    valid_config.q_limit = 500

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        input_ = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])
        label_ = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])
        train_bl = data_loader.TrainBatchLoader(train_images, input_, label_, train_config)
        valid_bl = data_loader.ValidBatchLoader(valid_images, input_, label_, valid_config)

        train_x, train_y = train_bl.get_batch()
        valid_x, valid_y = valid_bl.get_batch()

        device = ops.get_available_gpus()[0]
        with tf.device(device):

            with tf.variable_scope('Generator'):
                G = network.Generator(train_x, train_config)

            with tf.variable_scope('Discriminator'):
                D_fake = network.Discriminator(G.output, train_config)

            with tf.variable_scope('Discriminator', reuse=True):
                D_real = network.Discriminator(train_y, train_config)
            with tf.variable_scope('Generator', reuse=True):
                valid_G = network.Generator(valid_x, valid_config)

            with tf.variable_scope('Discriminator', reuse=True):
                valid_D_fake = network.Discriminator(valid_G.output, valid_config)
                valid_D_real = network.Discriminator(valid_y, valid_config)
                valid_D_fake_loss = tf.reduce_mean(tf.square(valid_D_fake.output))
                valid_D_real_loss = tf.reduce_mean(tf.square(1 - valid_D_real.output))

            valid_G_mse_loss = tf.reduce_mean(tf.abs(valid_y - valid_G.output))
            valid_G_tv_loss  = tf.reduce_mean(tf.image.total_variation(valid_G.output)) / (3*(patch_size ** 2))
            valid_G_dis_loss = tf.reduce_mean(tf.square(1 - valid_D_fake.output))
            valid_G_loss = valid_G_mse_loss + 0.02 * valid_G_tv_loss + train_config.lamda * valid_G_dis_loss

            D_fake_loss = tf.reduce_mean(tf.square(D_fake.output))
            D_real_loss = tf.reduce_mean(tf.square(1 - D_real.output))
            D_loss = D_fake_loss + D_real_loss

            G_mse_loss = tf.reduce_mean(tf.abs(train_y - G.output))
            G_tv_loss  = tf.reduce_mean(tf.image.total_variation(G.output)) / (3*(patch_size ** 2))
            G_dis_loss = tf.reduce_mean(tf.square(1 - D_fake.output))
            G_loss = G_mse_loss + 0.02 * G_tv_loss + train_config.lamda * G_dis_loss

            gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Generator')
            dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')

            G_train_step = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=gen_var_list)
            D_train_step = tf.train.AdamOptimizer(1e-5).minimize(D_loss, var_list=dis_var_list)

            tf.summary.image('train_HE', ycbcr2rgb(train_x))
            tf.summary.image('train_PAS', ycbcr2rgb(train_y))
            tf.summary.image('train_G_output', ycbcr2rgb(G.output))

            tf.summary.image('valid_HE_prev', ycbcr2rgb(valid_x))
            tf.summary.image('valid_HE_new', ycbcr2rgb(valid_y))

            tf.summary.image('valid_HE_prev_output', ycbcr2rgb(valid_G.output))
               
            tf.summary.scalar('D_fake_loss', D_fake_loss)
            tf.summary.scalar('D_real_loss', D_real_loss)
            tf.summary.scalar('D_loss', D_loss)
            tf.summary.scalar('G_dis_loss', G_dis_loss)
            tf.summary.scalar('G_mse_loss', G_mse_loss)
            tf.summary.scalar('G_tv_loss', G_tv_loss)
            tf.summary.scalar('G_loss', G_loss)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=0)

            merged = tf.summary.merge_all()

            model_name = 'Models'
            summary_step = 1000
            if os.path.exists("tensorboard/" + model_name):
                shutil.rmtree("tensorboard/" + model_name)
            train_writer = tf.summary.FileWriter("tensorboard/" + model_name, sess.graph)


            tf.train.start_queue_runners(sess=sess)
            train_bl.start_threads(sess, n_threads=train_config.n_threads)
            valid_bl.start_threads(sess, n_threads=valid_config.n_threads)
            for i in tqdm(range(30)): sleep(1)
            print(train_bl.queue.size().eval(), valid_bl.queue.size().eval())

            #Save loss information in a validation log.
            valid_log = open('valid_log.txt', 'w')

            n_eval_steps = valid_config.q_limit // valid_config.batch_size
            check = train_config.checkpoint
            min_loss = float('inf')
            start_time = time()
            for x in range(1,100):
                d_fake_loss, d_real_loss, g_loss = 0, 0, 0
                NumGen = max(3, int(7-x/4))
                for i in range(check):
                    for j in range(NumGen):
                        _, b = sess.run([G_train_step, G_loss])
                        g_loss += b
                    _, a1, a2 = sess.run([D_train_step, D_fake_loss, D_real_loss])
                    d_fake_loss += a1
                    d_real_loss += a2
                    if not i % summary_step:
                        summary_train = sess.run(merged)
                        train_writer.add_summary(summary_train, (x * check + i) * (1))
                res = np.mean([sess.run([valid_G_loss, valid_G_mse_loss, valid_G_tv_loss, valid_G_dis_loss, valid_D_fake_loss, valid_D_real_loss]) for _ in range(n_eval_steps)], axis=0)

                format_str = ('iter: %d valid_G_loss: %.3f valid_G_mse_loss: %.3f valid_G_tv_loss: %.3f valid_G_dis_loss: %.3f valid_D_fake_loss: %.3f valid_D_real_loss: %.3f train_G_loss: %.3f train_D_fake_loss: %.3f train_D_real_loss: %.3f time: %d')
                text = (format_str % (x*check, res[0], res[1], res[2], res[3], res[4], res[5], g_loss/(check*NumGen), d_fake_loss/check, d_real_loss/check, int(time()-start_time)))
                ops.print_out(valid_log, text)
                #Save the model.
                saver.save(sess, 'Models/{}'.format(x*check))
                if res[1] < min_loss:
                    min_loss = res[1]
                    #Save the model with the lowest validation L1 loss.
                    saver.save(sess, 'Models/best_model')
