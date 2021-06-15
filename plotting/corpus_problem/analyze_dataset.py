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


dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
Y = np.squeeze([d[0].tolist() for d in dataset])
print(Y.shape)

error_mat = np.zeros((50000, 50000))

# offset_Y = Y
# for o in range(50000):
#     print(o)
#     offset_Y = offset(offset_Y)
#     if o == 0:
#         error_list = RMSE_many(Y, offset_Y)
#     else:
#         error_list = RMSE_many(Y[:-o], offset_Y[:-o])
#     for j, error in enumerate(error_list):
#         i = j + o
#         error_mat[i, j] = error_mat[j, i] = error

# pd.DataFrame(error_mat).to_csv('error_mat_test.csv', index=False, header=None)
# np.save('error_mat_test.npy', error_mat, allow_pickle=False)

inf_indices = np.triu_indices(50)
print(inf_indices)
exit()
import h5py
with h5py.File('error_mat_test.hdf5', 'w') as f:
    dset = f.create_dataset('error_mat', data=error_mat)
