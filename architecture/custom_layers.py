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
        assert len(inputs.shape) == 3, 'GLU ERROR: Expected 3D input.'
        assert inputs.shape[2] == 2, 'GLU ERROR: Expected inputs.shape[2] == 2'
        A = inputs[:, :, 0]
        B = inputs[:, :, 1]
        return tf.math.multiply(A, tf.math.sigmoid(B))
