"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Read in equations_with_coeff for a specified dataset
         then recompute y-values multiple times for each equations
         on different x-values (picked from normal distributions centered
         in [0.1, 3.1)).

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize
from equation.EquationInfix import EquationInfix
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq
from srvgd.data_gathering.get_normal_x import get_normal_x

import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

import os


def rebuild_dataset_with_x(ff_list, num_obs, other_dataset_inputs=None):
    if other_dataset_inputs is None:
        other_dataset_inputs = []

    dataset_inputs = []
    dataset_outputs = []
    eq_list = []
    ff_index = 0
    while len(eq_list) < num_obs:
        ff = ff_list[ff_index]
        print(len(eq_list))
        print('ff', ff)
        skipped = False
        accepted = False
        coeff_attemps = 0
        while not accepted and not skipped:
            eq = EquationInfix(ff)
            X = get_normal_x(num_points=30)
            X.sort()
            coeffs = np.round(np.random.uniform(-3, 3, eq.num_coeffs), 3)
            Y = eq.f(coeffs, X.T)

            # Since I have checked this for the gridified x-values
            # when generating the dataset, I don't expect many (if any)
            # equations/x-value choices to be thrown out.
            # But, I'll check to be safe.
            if not np.any(np.isnan(Y)):
                if np.min(Y) != np.max(Y):
                    if np.all(np.abs(Y) <= 1000):
                        normalized_Y = np.around(normalize(Y), 7).tolist()
                        inp = np.vstack((X, normalized_Y)).T.tolist()
                        if inp not in dataset_inputs and inp not in other_dataset_inputs:
                            dataset_inputs.append(inp)
                            dataset_outputs.append(tokenize_eq(ff_list[ff_index]))
                            eq_list.append(eq.eq_str)
                            accepted = True
                            print('eq', eq.eq_str)
                            print('ACCEPTED!')
                            print('')

        coeff_attemps += 1
        print('.', flush=True, end='')
        if coeff_attemps > 100:
            skipped = True
            print('SKIPPED!')
            print('')

        ff_index = (ff_index+1) % len(ff_list)

    # pad dataset_outputs
    max_len = max([len(out) for out in dataset_outputs])
    dataset_outputs = [out + [0]*(max_len-len(out)) for out in dataset_outputs]

    return dataset_inputs, dataset_outputs, eq_list


if __name__ == '__main__':
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
    np.random.shuffle(ff_list)
    ff_list = ff_list[:1000]

    other_dataset_inputs = None
    for dataset_type in ['train', 'test']:
        num_obs = 50000 if dataset_type == 'train' else 1000
        dataset_parts = rebuild_dataset_with_x(ff_list, num_obs, other_dataset_inputs)

        dataset = TensorDataset(torch.Tensor(dataset_parts[0]), torch.LongTensor(dataset_parts[1]))
        torch.save(dataset, os.path.join(dataset_path, 'dataset_'+dataset_type+'_ff1000_normal_with_x.pt'))

        pd.DataFrame(dataset_parts[2]).to_csv(os.path.join(dataset_path, 'equations_with_coeff_'+dataset_type+'_ff1000_normal_with_x.csv'))
