"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 11, 2020

PURPOSE: Define new layers that I need to implement
         convolutional seq2seq model.

NOTES:

TODO: put attention in here.
"""

import tensorflow as tf
from tensorflow import keras


class GLU(keras.layers.Layer):
    def __init__(self, units=None, input_dim=None):
        super(GLU, self).__init__()

    def call(self, inputs):
        A = inputs[:, :, 0]
        B = inputs[:, :, 1]
        return tf.math.multiply(A, tf.math.sigmoid(B))
