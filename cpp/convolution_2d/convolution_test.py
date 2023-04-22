import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging

# uncomment the following line if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

x_in = np.array([[
  [[3], [1], [0], [2], [5], [6]],
  [[4], [2], [1], [1], [4], [7]],
  [[5], [4], [0], [0], [1], [2]],
  [[1], [2], [2], [1], [3], [4]],
  [[6], [3], [1], [0], [5], [2]],
  [[3], [1], [0], [1], [3], [3]], ]])

kernel_in = np.array([
 [ [[-1]], [[0]], [[1]] ],
 [ [[-1]], [[0]], [[1]] ],
 [ [[-1]], [[0]], [[1]] ], ])

x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)

result = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')

print(result)