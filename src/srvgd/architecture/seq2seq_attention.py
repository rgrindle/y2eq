from tensorflow import keras
from tensorflow.keras.layers import Dense, SimpleRNN, Input, dot

LATENTDIM = 256
INPUT_LENGTH = 1
NUM_OUTPUT_TOKENS = 21
MAX_OUTPUT_LENGTH = 13

encoder_inputs = Input(shape=((30, 1)))
encoder_outputs, encoder_state = SimpleRNN(LATENTDIM, return_state=True, return_sequences=True)(encoder_inputs)
print(encoder_outputs.shape, encoder_state.shape)
# current thinking tha tI don't need state, but let's wait

# get attention weights
# input last output and current encoder hidden state
decoder_inputs = Input(shape=((MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS)))
attention_outputs, attention_state = SimpleRNN(LATENTDIM, return_state=True, return_sequences=True)(decoder_inputs)
print(attention_outputs.shape, attention_state.shape)

score = dot([encoder_outputs, attention_outputs], axes=(2, 2))
print(score.shape)
alignment = keras.activations.softmax(score, axis=-1)
print(alignment.shape)
context = dot([alignment, encoder_outputs], axes=(1, 1))
print(context.shape)

dec_rec = SimpleRNN(LATENTDIM, return_sequences=True)(context)
dec_dense = Dense(NUM_OUTPUT_TOKENS, activation='softmax')(dec_rec)

model = keras.Model([encoder_inputs, decoder_inputs],
                    [dec_dense])


if __name__ == '__main__':
    import numpy as np

    encoder_inputs = np.zeros((2, 30, INPUT_LENGTH))
    decoder_inputs = np.zeros((2, MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS))
    outputs = model.predict([encoder_inputs, decoder_inputs])
    print(outputs.shape)
