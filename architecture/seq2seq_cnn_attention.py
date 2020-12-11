from custom_layers import GLU, Attention

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, SimpleRNN, Input, dot, Conv1D, Conv2D

LATENTDIM = 256
INPUT_LENGTH = 1
NUM_OUTPUT_TOKENS = DICTIONARY_LENGTH = 21
MAX_OUTPUT_LENGTH = 13
CONSISTENT_SIZE = 10

enc_inputs = Input(shape=((30, 1)))
print(enc_inputs.shape)
# TODO: positional encoding here
enc_conv_outputs = Conv1D(2, 3, padding='same')(enc_inputs)
print(enc_conv_outputs.shape)
enc_outputs = GLU()(enc_conv_outputs)
print(enc_outputs.shape)
enc_outputs = Dense(CONSISTENT_SIZE)(enc_outputs)
print(enc_outputs.shape)

print()
print('begin decoder')
# get attention weights
# input last output and current encoder hidden state
dec_inputs = Input(shape=((MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS)))
# TODO: positional encoding here
# TODO: masking. make sure doesn't see future outputs
atten_conv_outputs = Conv1D(2*NUM_OUTPUT_TOKENS, 3)(dec_inputs)
print(atten_conv_outputs.shape)
atten_outputs = GLU()(atten_conv_outputs)
print(atten_outputs.shape)
atten_outputs = Dense(CONSISTENT_SIZE)(atten_outputs)
print(atten_outputs.shape)

context = Attention()([enc_outputs, atten_outputs])
print(context.shape)

dec_dense = Dense(NUM_OUTPUT_TOKENS, activation='softmax')(context)

model = keras.Model([enc_inputs, dec_inputs],
                    [dec_dense])


if __name__ == '__main__':
    import numpy as np

    encoder_inputs = np.zeros((2, 30, INPUT_LENGTH))
    decoder_inputs = np.zeros((2, MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS))
    outputs = model.predict([encoder_inputs, decoder_inputs])
    print(outputs.shape)
