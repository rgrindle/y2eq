"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 12, 2021

PURPOSE: Take equation_with_coeffs_train_ff1000.csv and alter
         these equations to be two dimentional. Do the same for
         test equations.

NOTES: For every variable x incountered, change it to
       x0 (with 50% change) or x1 (with a 50% chance)

TODO:
"""
from srvgd.utils.normalize import normalize
from equation.EquationInfix import EquationInfix
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
    new_eq_list = []
    eq_index = 0
    while len(new_eq_list) < len(eq_list):
        eq = eq_list[eq_index]
        count = 0
        print(eq_index)
        print('eq', eq)
        ff = EquationInfix(eq, apply_coeffs=False).get_eq_no_coeff()
        ff = ff.replace('np.', '')

        print('ff', ff)

        if '.' in ff:
            print('STOPPING')
            exit()

        skipped = False
        accepted = False
        coeff_attemps = 0
        while not accepted and not skipped:
            new_eq = EquationInfix(ff)
            coeffs = np.round(np.random.uniform(-3, 3, new_eq.num_coeffs), 3)
            Y = new_eq.f(coeffs, X.T)

            if not np.any(np.isnan(Y)):
                if np.min(Y) != np.max(Y):
                    if np.all(np.abs(Y) <= 1000):
                        normalized_Y = np.around(normalize(Y), 7)
                        inp = np.hstack((X, normalized_Y.reshape(num_points, 1))).T.tolist()
                        if inp not in dataset_inputs:
                            count += 1
                            dataset_inputs.append(inp)
                            ff = ff.replace('x[0]', 'x0').replace('x[1]', 'x1')
                            dataset_outputs.append(tokenize_eq(ff, two_d=True))
                            accepted = True
                            print('ACCEPTED!')
                            print('')
                            new_eq_list.append(new_eq.place_exact_coeffs(coeffs))

            coeff_attemps += 1
            print('.', flush=True, end='')
            if coeff_attemps > 100:
                skipped = True
                print('SKIPPED!')
                print('')

        eq_index = (eq_index+1) % len(eq_list)

    # pad dataset_outputs
    max_len = max([len(out) for out in dataset_outputs])
    dataset_outputs = [out + [0]*(max_len-len(out)) for out in dataset_outputs]
    print(len(new_eq_list))
    return dataset_inputs, dataset_outputs, new_eq_list


if __name__ == '__main__':
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    for dataset_type in ['train', 'test']:
        eq_2d_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_{}_ff1000_2d.csv'.format(dataset_type)),
                                 header=None).values.flatten()

        dataset_parts = rebuild_dataset_with_x(eq_2d_list, num_points=1024)

        dataset = TensorDataset(torch.Tensor(dataset_parts[0]), torch.LongTensor(dataset_parts[1]))
        torch.save(dataset, os.path.join(dataset_path, 'dataset_{}_ff1000_2d.pt'.format(dataset_type)))

        pd.DataFrame(dataset_parts[2]).to_csv(os.path.join(dataset_path, 'equations_with_coeff_{}_ff1000_2d.csv'.format(dataset_type)),
                                              header=None,
                                              index=False)
