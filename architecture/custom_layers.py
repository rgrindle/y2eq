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
    def __init__(self, units, input_dim):
        super(GLU, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units),
                                 dtype='float32'),
            trainable=True,
        )

    def call(self, inputs):
        assert len(inputs) == 2, 'In GLU.call, expected inputs to be a list of length 2.'
        return tf.math.multipy(inputs[0], tf.math.sigmoid(inputs[1]))