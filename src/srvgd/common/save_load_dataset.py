"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 17, 2020

PURPOSE: Saving and loading datasets will happen
         in may places, so these function can be
         stored here for easy access.

NOTES:

TODO: Add save_dataset here. Am I really going to use that in multiple places?
"""
from srvgd.architecture.seq2seq_cnn_attention import MAX_OUTPUT_LENGTH, NUM_OUTPUT_TOKENS

import numpy as np

import os


token2onehot = {0: np.zeros(NUM_OUTPUT_TOKENS)}
for i in range(1, 23):
    vec = np.zeros(NUM_OUTPUT_TOKENS)
    vec[i-1] = 1.
    token2onehot[i] = vec

onehot2token = {tuple(value): key for key, value in token2onehot.items()}


def load_raw_dataset(path, dataset_type):
    dataset, info = np.load(path, allow_pickle=True)
    assert dataset_type in ('train', 'test')
    if dataset_type == 'train':
        assert info['isTraining']
    else:
        assert not info['isTraining']
    return dataset, info


def format_dataset(dataset):
    # get NN inputs
    x_dataset = [xy[0] for xy in dataset]
    x_arr = np.array([np.array(x) for x in x_dataset])[:, :, None]
    print('x shape', x_arr.shape)

    # get and then format NN outputs
    y_dataset = [xy[1] for xy in dataset]
    y_arr = np.array([np.array(y) for y in y_dataset])[:, :, None]

    # update padding if necessary
    if y_arr.shape[1] > MAX_OUTPUT_LENGTH+1:
        print('There are equations that are too long for MAX_OUTPUT_LENGTH in chosen model.')
    elif y_arr.shape[1] < MAX_OUTPUT_LENGTH+1:
        temp = np.zeros((y_arr.shape[0], MAX_OUTPUT_LENGTH+1, y_arr.shape[2]))
        temp[:, :y_arr.shape[1], :] = y_arr
        y_arr = temp

    # now get onehot version of arr_y
    onehot_y = np.array([[token2onehot[int(yi)] for yi in y] for y in y_arr])
    print('y shape', onehot_y.shape)

    y_input = onehot_y[:, :-1, :]  # remove last token
    y_target = onehot_y[:, 1:, :]  # remove START token
    return x_arr, y_input, y_target


def load_and_format_dataset(dataset_name, dataset_type, return_info=False):
    assert dataset_type in ('train', 'test')

    print('reading dataset ...', end='', flush=True)
    dataset, info = load_raw_dataset(os.path.join('..', '..', '..', 'datasets', '{}_{}.npy'.format(dataset_name, dataset_type)), dataset_type)
    print('done.')

    x, y_input, y_target = format_dataset(dataset)

    if return_info:
        return [x, y_input], y_target, info
    else:
        return [x, y_input], y_target
