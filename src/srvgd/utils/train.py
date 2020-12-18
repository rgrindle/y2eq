"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 17, 2020

PURPOSE: Train neural networks based on achitectures
         in the architecture folder and datasets generated
         with code in the data_gathering folder.

NOTES: 12 = start, 13 = stop

TODO:
"""
import numpy as np  # type: ignore
import pandas as pd  # type: ignoree
from tensorflow.keras.callbacks import ModelCheckpoint
from srvgd.architecture.seq2seq_cnn_attention import MAX_OUTPUT_LENGTH

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
                model_name, checkpoint=False):
    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('..', '..', '..', 'models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('..', '..', '..', 'models', 'weights_'+model_name),
                                 monitor='val_loss')

    if not checkpoint:
        model.compile(optimizer='adam', loss='categorical_crossentropy')
    # NOTE: y gets padded inside model as input.
    history = model.fit([x, y_input], y_target,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_cb, weights_cb])

    history_file = os.path.join('models', model_name+'_history.csv')
    if checkpoint:
        df = pd.read_csv(history_file)
        history_data = np.vstack((df.values, list(history.history.values())))
    else:
        history_data = history.history
    pd.DataFrame(history_data).to_csv(history_file, header=False, index=False)
    return model


def load_dataset(path, dataset_type):
    store_format = np.load(path, allow_pickle=True)
    dataset, info = store_format
    assert dataset_type in ('train', 'test')
    if dataset_type == 'train':
        assert info["isTraining"]
    else:
        assert not info["isTraining"]
    return dataset, info


def load_and_format_dataset(datset_type, return_info=False):
    assert datset_type in ('train', 'test')
    if datset_type == 'train':
        index = 0
    else:
        index = 2

    print('reading dataset ...', end='', flush=True)
    dataset, info = load_dataset(os.path.join('..', '..', '..', 'datasets', 'dataset_no_scaling_train.npy'), 'train')#[index:index+2]
    print('done.')

    # get NN inputs
    x_dataset = [xy[0] for xy in dataset]
    arr_x = np.array([np.array(x) for x in x_dataset])[:, :, None]
    print('x shape', arr_x.shape)

    # get and then format NN outputs
    y_dataset = [xy[1] for xy in dataset]
    arr_y = np.array([np.array(y) for y in y_dataset])[:, :, None]

    # update padding if necessary
    if arr_y.shape[1] > MAX_OUTPUT_LENGTH:
        print('There are equations that are too long for MAX_OUTPUT_LENGTH in chosen model.')
    elif arr_y.shape[1] < MAX_OUTPUT_LENGTH:
        temp = np.zeros((arr_y.shape[0], MAX_OUTPUT_LENGTH, arr_y.shape[2]))
        temp[:, :arr_y.shape[1], :] = arr_y
        arr_y = temp

    # now get onehot version of arr_y
    onehot_y = np.array([[token2onehot[int(yi)] for yi in y] for y in arr_y])
    print('y shape', onehot_y.shape)

    y_input = onehot_y[:, :-1, :]  # remove last token
    y_target = onehot_y[:, 1:, :]  # remove START token
    if return_info:
        return [arr_x, y_input], y_target, info
    else:
        return [arr_x, y_input], y_target


if __name__ == '__main__':
    from tensorflow import keras
    checkpoint = True
    if checkpoint:
        model = keras.models.load_model(os.path.join('..', '..', '..', 'models', 'model_seq2seq_cnn_attention_model'))

    else:
        from srvgd.architecture.seq2seq_cnn_attention import model

    x, y = load_and_format_dataset('train')

    trained_model = train_model(model, x[0], x[1], y,
                                batch_size=128,
                                epochs=200,
                                model_name='seq2seq_cnn_attention_model',
                                checkpoint=checkpoint)
