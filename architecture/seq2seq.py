"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 8, 2020

PURPOSE: This is a basic seq2seq model. It can be used
         as the achitecture in train.py

NOTES: This architecture has 1 layer in the encoder
       and one layer in the decoder. It uses LSTM's
       and does not use attention.

TODO:
"""

from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Input

LATENTDIM = 256
INPUT_LENGTH = 1
OUTPUT_LENGTH = 21

encoder_inputs = Input(shape=((None, INPUT_LENGTH)))
encoder_outputs, encoder_state_h, encoder_state_c = LSTM(LATENTDIM, return_state=True)(encoder_inputs)

decoder_inputs = Input(shape=((None, OUTPUT_LENGTH)))
dec_rec = LSTM(LATENTDIM, return_sequences=True)(decoder_inputs, initial_state=[encoder_state_h, encoder_state_c])
dec_dense = Dense(OUTPUT_LENGTH, activation='softmax')(dec_rec)
model = keras.Model([encoder_inputs, decoder_inputs],
                    [dec_dense])
