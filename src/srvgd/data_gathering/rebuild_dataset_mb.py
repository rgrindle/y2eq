"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 18, 2021

PURPOSE: Read in dataset_train_ff1000 and add m, b
         to the input sequence where m and b are the
         parameters in the line of best fit.

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize
from equation.EquationInfix import EquationInfix

import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

import os


def least_squares(x, y):
    """get m, b (and associated error)
    such that y = mx+b"""
    A = np.vstack((x, np.ones_like(x))).T
    coeffs, error, _, _ = np.linalg.lstsq(A, y, rcond=None)
    m, b = coeffs
    error = error[0]
    # import matplotlib.pyplot as plt
    # plt.plot(x, m*x+b, '-')
    # plt.plot(x, y, '.')
    # plt.show()
    return m, b, error


if __name__ == '__main__':
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    x = np.arange(0.1, 3.1, 0.1)
    eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000.csv'), header=None).values.flatten()
    eq_list = [EquationInfix(eq, apply_coeffs=False) for eq in eq_list]
    unnormalized_y_list = [eq.f(x) for eq in eq_list]

    mb_list = np.array([least_squares(x, y) for y in unnormalized_y_list])

    normalization_params = []
    normalized_values_list = []
    for i in range(mb_list.shape[1]):
        normalized_values, min_, scale_ = normalize(mb_list[:, i], return_params=True)
        normalized_values_list.append(normalized_values)
        normalization_params.append([min_, scale_])

    mb_list = np.array(normalized_values_list).T[:, None, :]

    dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                         map_location=torch.device('cpu'))

    dataset_input = np.array([d[0].tolist() for d in dataset])
    dataset_output = [d[1].tolist() for d in dataset]

    mb_list = np.repeat(mb_list, 30, axis=1)
    dataset_input = np.concatenate((dataset_input, mb_list), axis=-1)
    dataset_input = dataset_input.tolist()

    # pad dataset_output
    max_len = max([len(out) for out in dataset_output])
    padded_dataset_output = []
    for i, out in enumerate(dataset_output):
        dataset_output[i] = out+[0]*(max_len-len(out))

    print(len(dataset_input))
    print(len(dataset_output))

    dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
    torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_mb.pt'))

    pd.DataFrame(normalization_params).to_csv(os.path.join(dataset_path, 'mb_normalization_params.csv'),
                                              index=False,
                                              header=None)
