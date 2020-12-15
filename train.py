"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 13, 2020

PURPOSE: Train neural networks based on achitectures
         in the architecture folder and datasets generated
         with code in the data_gathering folder.

NOTES: 12 = start, 13 = stop

TODO:
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignoree
from tensorflow.keras.callbacks import ModelCheckpoint

import os


NUM_TOKENS = 22

token2onehot = {0: np.zeros(NUM_TOKENS)}
for i in range(1, 23):
    vec = np.zeros(NUM_TOKENS)
    vec[i-1] = 1.
    token2onehot[i] = vec

onehot2token = {tuple(value): key for key, value in token2onehot.items()}


def train_model(model, x, y_input, y_target,
                batch_size, epochs,
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


def load_dataset(path):
    store_format = np.load(path, allow_pickle=True)
    train_dataset, info_training, test_dataset, info_testing = store_format
    assert info_training["isTraining"]
    assert not info_testing["isTraining"]
    return train_dataset, info_training, test_dataset, info_testing


def load_and_format_dataset(datset_type, return_info=False):
    assert datset_type in ('train', 'test')
    if datset_type == 'train':
        index = 0
    else:
        index = 2

    print('reading dataset ...', end='', flush=True)
    dataset, info = load_dataset(os.path.join('datasets', 'dataset.npy'))[index:index+2]
    print('done.')
    x_dataset = [xy[0] for xy in dataset]
    y_dataset = [xy[1] for xy in dataset]
    arr_x = np.array([np.array(x) for x in x_dataset])[:, :, None]
    print('x shape', arr_x.shape)

    arr_y = np.array([np.array(y) for y in y_dataset])[:, :, None]
    onehot_y = np.array([[token2onehot[int(yi)] for yi in y] for y in arr_y])
    print('y shape', onehot_y.shape)

    y_input = onehot_y[:, :-1, :]  # remove last token
    y_target = onehot_y[:, 1:, :]  # remove START token
    if return_info:
        return [arr_x, y_input], y_target, info
    else:
        return [arr_x, y_input], y_target


if __name__ == '__main__':
    from architecture.seq2seq_cnn_attention import model
    # from eqlearner.dataset.processing import tokenization

    x, y = load_and_format_dataset('train')

    # dataset_file = 'dataset_maxdepth3_seed0_train.json'
    trained_model = train_model(model, x[0], x[1], y,
                                batch_size=128,
                                epochs=100,
                                model_name='seq2seq_cnn_attention_model')
    # result = eval_model(trained_model)
    # print(result)
