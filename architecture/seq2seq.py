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

if __name__ == '__main__':
    import numpy as np
    fake_input = np.zeros((10, 1, 30))
    output = model.predict([fake_input, output])
    print(output.shape)

