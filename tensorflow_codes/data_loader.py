import random, threading
import tensorflow as tf
import numpy as np
import scipy.io
import os.path as pathfun
from PIL import Image

class BatchLoader(object):

    def __init__(self, images, image_, label_, config):
        self.images = images
        self.image_ = image_
        self.label_ = label_
        self.config = config
        self.capacity = config.q_limit
        image_shape = self.image_.get_shape().as_list()[1:]
        label_shape = self.label_.get_shape().as_list()[1:]
        self.threads = []
        self.queue = tf.RandomShuffleQueue(shapes=[image_shape, label_shape],
                                            dtypes=[tf.float32, tf.float32],
                                            capacity=self.capacity,
                                            min_after_dequeue=0)
        self.enqueue_op = self.queue.enqueue_many([self.image_, self.label_])

    def get_batch(self):
        return self.queue.dequeue_many(self.config.batch_size)

    def create_thread(self, sess, thread_id, n_threads):
        for image_batch, label_batch in self.batch_generator(self.images[thread_id::n_threads]):
            sess.run(self.enqueue_op, feed_dict={self.image_: image_batch, self.label_: label_batch})

    def start_threads(self, sess, n_threads):
        for i in range(n_threads):
            thread = threading.Thread(target=self.create_thread, args=(sess, i, n_threads))
            self.threads.append(thread)
            thread.start()

class TrainBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)

    def batch_generator(self, paths):
        size = len(paths)
        s = self.config.image_size
        stride = s
        while True:
            for path in paths:
                path=path.replace('.tif', '.npy')
                rand_choice_path = random.randint(0, 8)
                if rand_choice_path == 0:
                    cyclegan_path='style0'
                elif rand_choice_path == 1:
                    cyclegan_path='style1'
                elif rand_choice_path == 2:
                    cyclegan_path='style2'
                elif rand_choice_path == 3:
                    cyclegan_path='style3'
                elif rand_choice_path == 4:
                    cyclegan_path='style4'
                elif rand_choice_path == 5:
                    cyclegan_path='style5'                
                elif rand_choice_path == 6:
                    cyclegan_path='style6'
                elif rand_choice_path == 7:
                    cyclegan_path='style7'
                elif rand_choice_path == 8:
                    cyclegan_path='original_style'

                image = np.load(path.replace('label',cyclegan_path))
                image_temp = image.copy()
                image[:,:,0] = 0.299*image_temp[:,:,0]+0.587*image_temp[:,:,1]+0.114*image_temp[:,:,2]
                image[:,:,1] = (image_temp[:,:,0] - image[:,:,0])*0.713 + 128
                image[:,:,2] = (image_temp[:,:,2] - image[:,:,0])*0.564 + 128

                label = np.load(path)
                label_temp = label.copy()
                label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
                label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 + 128
                label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 + 128


                size = image.shape[0]
                images, labels = [], []
                x = 0
                while True:
                    y = 0
                    while True:
                        rand_choice_stride=random.randint(0, 15)
                        xx = min(x+rand_choice_stride*s//16, size - s)
                        yy = min(y+rand_choice_stride*s//16, size - s)
                        if yy != size - s and xx != size - s:
                            img = image[xx:xx+s, yy:yy+s,:]
                            lab = label[xx:xx+s, yy:yy+s,:]
                            temp_lab = label_temp[xx:xx+s, yy:yy+s,:]
                            if np.mean(temp_lab) < 242:
                                rand_choice=random.randint(0, 5)

                                if rand_choice==0:

                                       img = np.fliplr(img)
                                       lab = np.fliplr(lab)
                                elif rand_choice==1:
                                       img = np.flipud(img)
                                       lab = np.flipud(lab)
                                elif rand_choice==2:
                                       img = np.rot90(img, k=1)
                                       lab = np.rot90(lab, k=1)
                                elif rand_choice==3:
                                       img = np.rot90(img, k=2)
                                       lab = np.rot90(lab, k=2)
                                elif rand_choice==4:
                                       img = np.rot90(img, k=3)
                                       lab = np.rot90(lab, k=3)
                                elif rand_choice==5:
                                       img = img
                                       lab = lab
                                       
                                images.append(img)
                                labels.append(lab)

                        if yy == size - s:
                            break
                        y += stride
                    if xx == size - s:
                        break
                    x += stride
                try:
                    images[0].shape
                    yield np.array(images), np.array(labels)
                except IndexError:
                    print('Training image loading error')

class ValidBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)

    def batch_generator(self, paths):
        size = len(paths)
        s = self.config.image_size
        stride = s
        while True:
            images, labels = [], []
            for path in paths:

                path=path.replace('.tif', '.npy')
                rand_choice_path = random.randint(0, 8)
                if rand_choice_path == 0:
                    cyclegan_path='style0'
                elif rand_choice_path == 1:
                    cyclegan_path='style1'
                elif rand_choice_path == 2:
                    cyclegan_path='style2'
                elif rand_choice_path == 3:
                    cyclegan_path='style3'
                elif rand_choice_path == 4:
                    cyclegan_path='style4'
                elif rand_choice_path == 5:
                    cyclegan_path='style5'                
                elif rand_choice_path == 6:
                    cyclegan_path='style6'
                elif rand_choice_path == 7:
                    cyclegan_path='style7'
                elif rand_choice_path == 8:
                    cyclegan_path='original_style'


                image = np.load(path.replace('label',cyclegan_path))
                image_temp = image.copy()
                image[:,:,0] = 0.299*image_temp[:,:,0]+0.587*image_temp[:,:,1]+0.114*image_temp[:,:,2]
                image[:,:,1] = (image_temp[:,:,0] - image[:,:,0])*0.713 + 128
                image[:,:,2] = (image_temp[:,:,2] - image[:,:,0])*0.564 + 128

                label = np.load(path)
                label_temp = label.copy()
                label[:,:,0] = 0.299*label_temp[:,:,0]+0.587*label_temp[:,:,1]+0.114*label_temp[:,:,2]
                label[:,:,1] = (label_temp[:,:,0] - label[:,:,0])*0.713 + 128
                label[:,:,2] = (label_temp[:,:,2] - label[:,:,0])*0.564 + 128

                size = image.shape[0]-1
                x = 0
                while True:
                    y = 0
                    while True:
                        xx = min(x, size - s)
                        yy = min(y, size - s)
                        images.append(image[xx:xx+s, yy:yy+s,:])
                        labels.append(label[xx:xx+s, yy:yy+s,:])
                        if yy == size - s:
                            break
                        y += stride
                    if xx == size - s:
                        break
                    x += stride
            # print(len(images))
            # exit()
            try:
                images[0].shape
                yield np.array(images), np.array(labels)
            except IndexError:
                print('Validation image loading error')

class TestBatchLoader(object):
    def __init__(self, images, image_):
        self.images = images
        self.image_ = image_
        image_shape = self.image_.get_shape().as_list()[1:]
        self.queue = tf.FIFOQueue(shapes=[image_shape], dtypes=[tf.float32], capacity=10)
        self.enqueue_op = self.queue.enqueue_many([self.image_])

    def create_thread(self, sess):
        for x in self.images:
            sess.run(self.enqueue_op, feed_dict={self.image_:  np.expand_dims(np.expand_dims(np.load(x), axis=0), axis = 3)})

    def start_thread(self, sess):
        thread = threading.Thread(target=self.create_thread, args=(sess,))
        thread.start()
