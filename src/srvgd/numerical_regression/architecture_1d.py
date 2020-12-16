"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 16, 2020

PURPOSE: The achitecture can be trained to
         approximate a function f: R -> R.

NOTES:

TODO:
"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

INPUT_DIMS = 1
HIDDEN_LAYER_SIZE = 100

inputs = Input(shape=(INPUT_DIMS,))
hidden_outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(inputs)
hidden_outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_outputs)
hidden_outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_outputs)
outputs = Dense(1)(hidden_outputs)

model = Model(inputs, outputs)

if __name__ == '__main__':
    import numpy as np

    inputs = np.array([[1], [1.1], [1.2]])
    outputs = model.predict(inputs)
    print(outputs)
