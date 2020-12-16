"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 15, 2020

PURPOSE: Reimplementation of architecture found in

         Biggio, Luca, Tommaso Bendinelli, Aurelien Lucchi,
         and Giambattista Parascandolo. "A Seq2Seq approach
         to Symbolic Regression."

         This version uses a loop because the target is not
         known. In other words, there is no teacher forcing.

NOTES: Not sure on some details like CONSISTENT_SIZE,
       kernel size in conv layers, positional encoding
       used.

TODO:
"""

from srvgd.architecture.custom_layers import GLU, Attention, PositionalEncoding

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, ZeroPadding1D, Add, concatenate

INPUT_LENGTH = 1
NUM_OUTPUT_TOKENS = DICTIONARY_LENGTH = 22
MAX_OUTPUT_LENGTH = 95
CONSISTENT_SIZE = 100

enc_inputs = Input(shape=((30, 1)), name='enc_input')
enc_inputs_with_pos_enc = PositionalEncoding()(enc_inputs)

enc_inputs_with_pos_enc_resized = Dense(CONSISTENT_SIZE)(enc_inputs_with_pos_enc)
enc_conv_outputs = Conv1D(2, 3, padding='same')(enc_inputs_with_pos_enc_resized)
enc_outputs = GLU()(enc_conv_outputs)
enc_outputs = Dense(CONSISTENT_SIZE)(enc_outputs)

enc_residual = Add()([enc_outputs, enc_inputs_with_pos_enc_resized])

dec_inputs = Input(shape=((1, NUM_OUTPUT_TOKENS)),
                   name='dec_input')
dec_inputs_padded = ZeroPadding1D((2, 0))(dec_inputs)

# define remaining decoder layers (not skip to output)
dec_inputs_with_pos_enc_padded_layer = PositionalEncoding()

dec_conv_outputs_layer = Conv1D(2*NUM_OUTPUT_TOKENS, 3)
dec_glu_outputs_layer = GLU()
dec_glu_dense_outputs_layer = Dense(CONSISTENT_SIZE)

context_layer = Attention()

dec_residual_layer = Add()

dec_outputs_layer = Dense(NUM_OUTPUT_TOKENS, activation='softmax')

all_outputs = []
while len(all_outputs) < MAX_OUTPUT_LENGTH:
    dec_inputs_with_pos_enc_padded = dec_inputs_with_pos_enc_padded_layer(dec_inputs_padded)
    dec_conv_outputs = dec_conv_outputs_layer(dec_inputs_with_pos_enc_padded)
    dec_glu_outputs = dec_glu_outputs_layer(dec_conv_outputs)
    dec_glu_dense_outputs = dec_glu_dense_outputs_layer(dec_glu_outputs)
    context = context_layer([enc_outputs, dec_glu_dense_outputs, enc_residual])
    dec_residual = dec_residual_layer([context, dec_glu_dense_outputs])
    dec_outputs = dec_outputs_layer(dec_residual)

    all_outputs.append(dec_outputs)
    dec_inputs_padded = concatenate([dec_inputs_padded, dec_outputs], axis=1)[:, -3:]

model = keras.Model([enc_inputs, dec_inputs],
                    all_outputs)


def predict_model(enc_inputs, dec_outputs):
    outputs = model.predict([encoder_inputs, dec_outputs])
    return np.moveaxis(np.squeeze(outputs), 0, 1)


if __name__ == '__main__':
    import numpy as np

    BATCHSIZE = 2
    START_TOKEN = np.zeros((1, NUM_OUTPUT_TOKENS))
    START_TOKEN[0, 13] = 1.

    encoder_inputs = np.random.uniform(-1, 1, size=(BATCHSIZE, 30, INPUT_LENGTH))
    decoder_inputs = np.repeat(START_TOKEN, BATCHSIZE, axis=0)[:, None]

    outputs = predict_model(encoder_inputs, decoder_inputs)
    print(len(outputs))
    print(np.array(outputs).shape)
