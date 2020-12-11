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
        half = inputs.shape[2]//2
        assert inputs.shape[2]/2 == half, 'GLU ERROR: Expected inputs.shape[2] to be divisible by 2.'
        A = inputs[:, :, :half]
        B = inputs[:, :, half:]
        return tf.math.multiply(A, tf.math.sigmoid(B))


class Attention(keras.layers.Layer):
    def __init__(self, units=None, input_dim=None):
        super(Attention, self).__init__()

    def call(self, inputs):
        """Expects inputs = [enc_outputs, atten_outputs]"""
        assert len(inputs) == 2
        assert inputs[0].shape[2] == inputs[1].shape[2]
        score = keras.layers.dot(inputs, axes=(2, 2))
        print('score', score.shape)
        alignment = keras.activations.softmax(score, axis=-1)
        print('alignment', alignment.shape)
        context = keras.layers.dot([alignment, inputs[0]], axes=(1, 1))
        print('Not quite right need to add something')  # TODO
        print('context', context.shape)
        return context