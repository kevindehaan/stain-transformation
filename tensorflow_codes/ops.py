from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import device_lib
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys

def evaluate(sess, n_steps, tf_ops):
    res = [sess.run(tf_ops) for _ in tqdm(range(n_steps))]
    return np.mean(res, axis=0)

def average_gradients(tower_grads):
    average_grads = []
    for gv in zip(*tower_grads):
        grads = [tf.expand_dims(g,0) for g, _ in gv]
        grad = tf.reduce_mean(tf.concat(grads, 0), 0)
        average_grads.append((grad, gv[0][1]))
    return average_grads

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def print_out(file, text):
    file.write(text + '\n')
    file.flush()
    print(text)
    sys.stdout.flush()
