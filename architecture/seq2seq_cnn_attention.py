from custom_layers import GLU, Attention

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, ZeroPadding1D, Add

LATENTDIM = 256
INPUT_LENGTH = 1
NUM_OUTPUT_TOKENS = DICTIONARY_LENGTH = 21
MAX_OUTPUT_LENGTH = 13
CONSISTENT_SIZE = 10

enc_inputs = Input(shape=((30, 1)), name='enc_input')
enc_inputs_resized = Dense(CONSISTENT_SIZE)(enc_inputs)

print(enc_inputs_resized.shape)
# TODO: positional encoding here
enc_conv_outputs = Conv1D(2, 3, padding='same')(enc_inputs_resized)
print(enc_conv_outputs.shape)
enc_outputs = GLU()(enc_conv_outputs)
print(enc_outputs.shape)
enc_outputs = Dense(CONSISTENT_SIZE)(enc_outputs)
print(enc_outputs.shape)

enc_residual = Add()([enc_outputs, enc_inputs_resized])
print(enc_residual.shape)

print()
print('begin decoder')
# get attention weights
# input last output and current encoder hidden state
dec_inputs = Input(shape=((MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS)),
                   name='dec_input')
print(dec_inputs.shape)
dec_inputs_padded = ZeroPadding1D((2, 0))(dec_inputs)
print(dec_inputs_padded.shape)
print(dec_inputs_padded)
# TODO: positional encoding here
dec_conv_outputs = Conv1D(2*NUM_OUTPUT_TOKENS, 3)(dec_inputs_padded)
print(dec_conv_outputs.shape)
dec_glu_outputs = GLU()(dec_conv_outputs)
print(dec_glu_outputs.shape)
dec_glu_outputs = Dense(CONSISTENT_SIZE)(dec_glu_outputs)
print(dec_glu_outputs.shape)

context = Attention()([enc_outputs, dec_glu_outputs, enc_residual])
print(context.shape)

dec_residual = Add()([context, dec_glu_outputs])
print(dec_residual.shape)

dec_outputs = Dense(NUM_OUTPUT_TOKENS, activation='softmax')(dec_residual)

model = keras.Model([enc_inputs, dec_inputs],
                    [dec_outputs])
exit()
if __name__ == '__main__':
    import numpy as np

    encoder_inputs = np.zeros((2, 30, INPUT_LENGTH))
    decoder_inputs = np.zeros((2, MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS))
    outputs = model.predict([encoder_inputs, decoder_inputs])
    print(outputs.shape)
