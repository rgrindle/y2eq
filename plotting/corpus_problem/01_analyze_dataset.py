"""
AUTHOR: Ryan Grindle

LAST MODIFIED: June 15, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES:

TODO:
"""
import numpy as np
import pandas as pd
import torch


def RMSE_many(Y, Y_hat):
    assert Y.shape == Y_hat.shape
    assert len(Y.shape) == 2
    return np.sqrt(np.mean(np.power(np.subtract(Y, Y_hat), 2), axis=1))


def offset(data):
    """offset by one

    If data = {d_i},
    then do d_{i+1} -> d_i
    """
    return np.array([data[(i+1) % len(data)] for i, x in enumerate(data)])


min_list = np.array([np.inf]*100)
ind_list = np.array([None]*100)

i_max = np.argmax(min_list)


def check_if_min(index, value):
    global min_list, ind_list, i_max

    if value < min_list[i_max]:
        min_list[i_max] = value
        ind_list[i_max] = index
        i_max = np.argmax(min_list)


dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
Y = np.squeeze([d[0].tolist() for d in dataset])
print(Y.shape)

offset_Y = Y
for o in range(50000):
    print(o)
    offset_Y = offset(offset_Y)
    if o == 0:
        error_list = RMSE_many(Y, offset_Y)
    else:
        error_list = RMSE_many(Y[:-o], offset_Y[:-o])
    for j, error in enumerate(error_list):
        i = j + o
        check_if_min(index=(i, j), value=error)

pd.DataFrame(min_list).to_csv('min_list.csv', index=False, header=None)
pd.DataFrame(ind_list).to_csv('ind_list.csv', index=False, header=None)
