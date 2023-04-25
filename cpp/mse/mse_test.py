import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging

# uncomment the following line if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

mse = tf.keras.losses.MeanSquaredError()

y_in = np.array([
  [1, 2],
  [0, 1],
  [-1, 1],
  [-2, 3]
])

t_in = np.array([
  [0.5, 2],
  [0.5, 1.5],
  [-2.5, 1.0],
  [-3.5, 3.5]
])

y_true = tf.constant(y_in, dtype=tf.float32)
y_pred = tf.constant(t_in, dtype=tf.float32)

# y_true = [[0., 1.], [0., 0.]]
# y_pred = [[1., 1.], [1., 0.]]

# 0.1125
# y_true = [[1.9], [-1.], [3.5], [0.0]]
# y_pred = [[2.0], [-1.2], [4.1], [0.2]]

result = mse(y_true, y_pred).numpy()

print(result)

y_true = [
  [[1., 2., 1.], [-3., 0, 2.]],
  [[5., -1., 3.], [1., 0.5, -1.5]],
  [[-2., -2., 1.], [1., -1., 1.]],
  [[-2., 0., 1.], [-1., -1., 3.]]
]

y_pred = [
    [[0.5, 2., 1.], [1., 1., 2.]],
    [[4., -2., 2.5], [0.5, 1.5, -2.]],
    [[-2.5, -2.8, 0.], [1.5, -1.2, 1.8]],
    [[-3., 1., -1.], [-1., -1., 3.5]]
]

result = mse(y_true, y_pred).numpy()

print(result)