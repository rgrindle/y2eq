"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 12, 2021

PURPOSE: Take equation_with_coeffs_train_ff1000.csv and alter
         these equations to be two dimentional.

NOTES: For every variable x incountered, change it to
       x0 (with 50% change) or x1 (with a 50% chance)

TODO:
"""
from srvgd.utils.normalize import normalize
from equation.EquationInfix import EquationInfix
from srvgd.utils.eval import get_f
from get_2d_grid import get_2d_grid
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq

import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

import os


def rebuild_dataset_with_x(eq_list, num_points):
    X = get_2d_grid(num_points)
    print(X)

    dataset_inputs = []
    dataset_outputs = []
    for eq_index, eq in enumerate(eq_list):
        count = 0
        print(eq_index)
        print('eq', eq)
        f = get_f(eq)
        Y = f(X.T)
        ff = EquationInfix(eq, apply_coeffs=False).get_eq_no_coeff()
        for p in ['sin', 'log', 'exp']:
            ff = ff.replace('np.'+p, p)
        ff = ff.replace('x[0]', 'x0').replace('x[1]', 'x1')
        print('ff', ff)
        print('')

        if '.' in ff:
            print('STOPPING')
            exit()

        if not np.any(np.isnan(Y)):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    normalized_Y = np.around(normalize(Y), 7)
                    inp = np.hstack((X, normalized_Y.reshape(num_points, 1))).T.tolist()
                    if inp not in dataset_inputs:
                        count += 1
                        dataset_inputs.append(inp)
                        dataset_outputs.append(tokenize_eq(ff))
                        print('.', flush=True, end='')

    # pad dataset_outputs
    max_len = max([len(out) for out in dataset_outputs])
    dataset_outputs = [out + [0]*(max_len-len(out)) for out in dataset_outputs]

    return dataset_inputs, dataset_outputs


if __name__ == '__main__':
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    eq_2d_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000_2d.csv'),
                             header=None).values.flatten()

    dataset_parts = rebuild_dataset_with_x(eq_2d_list, num_points=1024)

    dataset = TensorDataset(torch.Tensor(dataset_parts[0]), torch.LongTensor(dataset_parts[1]))
    torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_2d.pt'))
