"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 17, 2020

PURPOSE: Reimplementation of architecture found in

         Biggio, Luca, Tommaso Bendinelli, Aurelien Lucchi,
         and Giambattista Parascandolo. "A Seq2Seq approach
         to Symbolic Regression."

NOTES: Not sure on some details like CONSISTENT_SIZE,
       kernel size in conv layers, positional encoding
       used.

TODO:
"""

from srvgd.architecture.custom_layers import GLU, Attention, PositionalEncoding

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, ZeroPadding1D, Add

INPUT_LENGTH = 1
NUM_OUTPUT_TOKENS = DICTIONARY_LENGTH = 22
CONSISTENT_SIZE = 100
MAX_OUTPUT_LENGTH = 150


def get_model():
    enc_inputs = Input(shape=((30, 1)), name='enc_input')
    enc_inputs_with_pos_enc = PositionalEncoding()(enc_inputs)

    enc_inputs_with_pos_enc_resized = Dense(CONSISTENT_SIZE)(enc_inputs_with_pos_enc)
    enc_conv_outputs = Conv1D(2, 3, padding='same')(enc_inputs_with_pos_enc_resized)
    enc_outputs = GLU()(enc_conv_outputs)
    enc_outputs = Dense(CONSISTENT_SIZE)(enc_outputs)

    enc_residual = Add()([enc_outputs, enc_inputs_with_pos_enc_resized])

    dec_inputs = Input(shape=((MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS)),
                       name='dec_input')
    dec_inputs_padded = ZeroPadding1D((2, 0))(dec_inputs)
    dec_inputs_with_pos_enc_padded = PositionalEncoding()(dec_inputs_padded)

    dec_conv_outputs = Conv1D(2*NUM_OUTPUT_TOKENS, 3)(dec_inputs_with_pos_enc_padded)
    dec_glu_outputs = GLU()(dec_conv_outputs)
    dec_glu_outputs = Dense(CONSISTENT_SIZE)(dec_glu_outputs)

    context = Attention()([enc_outputs, dec_glu_outputs, enc_residual])

    dec_residual = Add()([context, dec_glu_outputs])

    dec_outputs = Dense(NUM_OUTPUT_TOKENS, activation='softmax')(dec_residual)

    model = keras.Model([enc_inputs, dec_inputs],
                        [dec_outputs])
    return model


if __name__ == '__main__':
    import numpy as np

    model = get_model()

    encoder_inputs = np.zeros((2, 30, INPUT_LENGTH))
    decoder_inputs = np.zeros((2, MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS))
    outputs = model.predict([encoder_inputs, decoder_inputs])
    print(outputs.shape)
