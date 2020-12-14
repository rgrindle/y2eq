"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 13, 2020

PURPOSE: Train neural networks based on achitectures
         in the architecture folder and datasets generated
         with code in the data_gathering folder.

NOTES: 12 = start, 13 = stop

TODO:
"""
from eqlearner.dataset.utils import load_dataset


import numpy as np  # type: ignore
import pandas as pd  # type: ignoree
from tensorflow.keras.callbacks import ModelCheckpoint

import os


def train_model(model, x, y, batch_size, epochs,
                model_name):
    # encoder_inputs, decoder_targets = load_dataset(dataset_file)
    # encoder_inputs = encoder_inputs[:, :, None]
    # print(encoder_inputs.shape, decoder_targets.shape)

    # indices = [i for i, x in enumerate(np.max(np.abs(encoder_inputs), axis=1)) if x < 1000]
    # encoder_inputs = encoder_inputs[indices]
    # decoder_targets = decoder_targets[indices]
    # decoder_inputs = get_decoder_inputs(decoder_targets)
    # print(encoder_inputs.shape, decoder_targets.shape)
    # print('nans?', np.any(np.isnan(encoder_inputs)), np.any(np.isnan(decoder_targets)))
    # print('infs?', np.any(np.isinf(encoder_inputs)), np.any(np.isinf(decoder_targets)))
    # print('max?', np.max(encoder_inputs), np.max(decoder_targets))

    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('models', 'weights_'+model_name),
                                 monitor='val_loss')

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # NOTE: y gets padded inside model as input.
    y_input = y[:, :-1, :]  # remove last token
    y_target = y[:, 1:, :]  # remove START tokens
    history = model.fit([x, y_input], y_target,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_cb, weights_cb])
    pd.DataFrame(history.history).to_csv(os.path.join('models', model_name+'_history.csv'), header=False, index=False)
    return model


# def get_decoder_inputs(decoder_targets):
#     # get decoder inputs (use teacher forcing)
#     decoder_inputs = np.zeros_like(decoder_targets)
#     print(decoder_inputs.shape, decoder_targets.shape)

#     # put START token at the beginning and shift right
#     # everything else by one.
#     decoder_inputs[:, 0] = onehot_encoder.transform([['START']]).toarray()
#     decoder_inputs[:, 1:] = decoder_targets[:, :-1]
#     return decoder_inputs


def eval_model(model):
    # output = trained_model.predict([encoder_inputs, decoder_inputs])
    # print(output.shape)
    pass


if __name__ == '__main__':
    from architecture.seq2seq_cnn_attention import model
    # from eqlearner.dataset.processing import tokenization

    NUM_TOKENS = 22

    token2onehot = {0: np.zeros(NUM_TOKENS)}
    for i in range(1, 23):
        vec = np.zeros(NUM_TOKENS)
        vec[i-1] = 1.
        token2onehot[i] = vec

    dataset_train = load_dataset(os.path.join('datasets', 'dataset_train_and_test.npy'))[0]
    x_dataset_train = [xy[0] for xy in dataset_train]
    y_dataset_train = [xy[1] for xy in dataset_train]
    arr_x = np.array([np.array(x) for x in x_dataset_train])[:, :, None]
    print('x shape', arr_x.shape)

    arr_y = np.array([np.array(y) for y in y_dataset_train])[:, :, None]
    onehot_y = np.array([[token2onehot[int(yi)] for yi in y] for y in arr_y])
    print('y shape', onehot_y.shape)

    # dataset_file = 'dataset_maxdepth3_seed0_train.json'
    trained_model = train_model(model, arr_x, onehot_y,
                                batch_size=128,
                                epochs=50,
                                model_name='seq2seq_cnn_attention_model')
    # result = eval_model(trained_model)
    # print(result)
