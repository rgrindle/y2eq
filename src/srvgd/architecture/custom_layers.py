"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 11, 2020

PURPOSE: Define new layers that I need to implement
         convolutional seq2seq model.

NOTES:

TODO: put attention in here.
"""

import numpy as np
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
        """Expects inputs = [enc_outputs, atten_outputs, enc_residual]"""
        assert len(inputs) == 3
        assert inputs[0].shape[2] == inputs[1].shape[2] == inputs[2].shape[2]
        score = keras.layers.dot(inputs[:2], axes=(2, 2))
        alignment = keras.activations.softmax(score, axis=-1)
        return keras.layers.dot([alignment, tf.math.add(inputs[0], inputs[2])], axes=(1, 1))


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, units=None, input_dim=None):
        super(PositionalEncoding, self).__init__()

    def call(self, inputs):
        positions = np.arange(inputs.shape[1], dtype=np.float32)
        vector_locations = np.arange(inputs.shape[2], dtype=np.float32)
        d = inputs.shape[2]
        denoms = 1000.**(2./d*vector_locations)
        denoms = np.repeat(denoms[None, :], repeats=inputs.shape[1], axis=0)
        numerators = np.repeat(positions[:, None], repeats=inputs.shape[2], axis=1)
        angles = (numerators/denoms)[None]
        positional_encodings = np.zeros_like(angles)
        positional_encodings[:, :, 0::2] = np.sin(angles[:, :, 0::2])
        positional_encodings[:, :, 1::2] = np.cos(angles[:, :, 1::2])
        return tf.math.add(inputs, tf.convert_to_tensor(positional_encodings))
