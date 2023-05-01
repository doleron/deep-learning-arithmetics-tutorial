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

x = tf.Variable(x_in, dtype=tf.float32)
kernel = tf.Variable(kernel_in, dtype=tf.float32)

with tf.GradientTape() as tape:
    y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    loss = tf.reduce_sum(y**2)

grad = tape.gradient(loss, kernel)

print("\nx\n", tf.squeeze(x).numpy())
print("\nkernel\n", tf.squeeze(kernel).numpy())
print("\ny\n", tf.squeeze(y).numpy())
print("\nloss\n", tf.squeeze(loss).numpy())
print("\ngrad\n", tf.squeeze(grad).numpy())
