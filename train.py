"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 8, 2020

PURPOSE: Train neural networks based on achitectures
         in the architecture folder and datasets generated
         with code in the data_gathering folder.

NOTES:

TODO:
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignoree
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder  # type: ignore

import os
import json

# get onehot encoder/decoder prepared
num_args = {'*': 2, '+': 2, 'sin': 1, 'log': 1, 'exp': 1}
tokens = list(num_args.keys())
tokens += ['x0', '(', ')', '^']
tokens += [str(d) for d in range(10)]
tokens += ['START', 'STOP']
onehot_encoder = OneHotEncoder().fit([[t] for t in tokens])


def load_dataset(save_name: str,
                 save_loc: str = os.path.join('datasets')):
    dataset_file = open(os.path.join(save_loc, save_name), 'r')
    dataset_inputs, dataset_outputs = json.load(dataset_file)
    return np.array(dataset_inputs), np.array(dataset_outputs)


def decode_train_data(onehot):
    eq_list = onehot_encoder.inverse_transform(onehot).flatten()
    return ''.join([x for x in eq_list if x != 'STOP'])


def train_model(model, dataset_file, batch_size, epochs,
                model_name):
    encoder_inputs, decoder_targets = load_dataset(dataset_file)
    encoder_inputs = encoder_inputs[:, :, None]
    print(encoder_inputs.shape, decoder_targets.shape)

    indices = [i for i, x in enumerate(np.max(np.abs(encoder_inputs), axis=1)) if x < 1000]
    encoder_inputs = encoder_inputs[indices]
    decoder_targets = decoder_targets[indices]
    decoder_inputs = get_decoder_inputs(decoder_targets)
    print(encoder_inputs.shape, decoder_targets.shape)
    print('nans?', np.any(np.isnan(encoder_inputs)), np.any(np.isnan(decoder_targets)))
    print('infs?', np.any(np.isinf(encoder_inputs)), np.any(np.isinf(decoder_targets)))
    print('max?', np.max(encoder_inputs), np.max(decoder_targets))

    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('models', 'weights_'+model_name),
                                 monitor='val_loss')

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    history = model.fit([encoder_inputs, decoder_inputs], decoder_targets,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_cb, weights_cb])
    pd.DataFrame(history.history).to_csv(os.path.join('models', model_name+'_history.csv'), header=False, index=False)
    return model


def get_decoder_inputs(decoder_targets):
    # get decoder inputs (use teacher forcing)
    decoder_inputs = np.zeros_like(decoder_targets)
    print(decoder_inputs.shape, decoder_targets.shape)

    # put START token at the beginning and shift right
    # everything else by one.
    decoder_inputs[:, 0] = onehot_encoder.transform([['START']]).toarray()
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    return decoder_inputs


def eval_model(model):
    # output = trained_model.predict([encoder_inputs, decoder_inputs])
    # print(output.shape)
    pass


if __name__ == '__main__':
    from architecture.seq2seq import model

    dataset_file = 'dataset_maxdepth3_seed0_train.json'
    trained_model = train_model(model, dataset_file,
                                batch_size=128,
                                epochs=100,
                                model_name='seq2seq_model')
    # result = eval_model(trained_model)
    # print(result)
