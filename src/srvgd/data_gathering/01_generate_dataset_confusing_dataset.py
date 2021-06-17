"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 15, 2021

PURPOSE: Take a list of functional forms and generate
         a dataset of a given size from them. Inputs will
         be y-values and outputs will be the functional forms.
         I expect that examples of different functional forms
         that result in similar y-values will be more difficult
         for the NN to learn from. To test this, I am making
         a dataset that has many similar y-values.

NOTES: In 01 (this script), create a fourth of the
       dataset by using a fourth of the functional forms
       and creating observation for the dataset by choicing
       coefficients randomly.

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq
from srvgd.utils.normalize import normalize

import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

import os


def get_one_coeff(rand_interval):
    return np.around(np.random.uniform(*rand_interval), 3)


def get_all_coeffs(num_coeffs, rand_interval):
    coeffs = [get_one_coeff(rand_interval) for _ in range(num_coeffs)]
    return coeffs


def get_dataset(x, ff_list, dataset_size,
                other_dataset_inputs=None):
    if other_dataset_inputs is None:
        other_dataset_inputs = []

    ff_list = list(ff_list)

    dataset_inputs = []
    dataset_outputs = []
    eq_with_coeff_list = []
    count = 0
    attempt_count = 0
    while count < dataset_size:
        ff = ff_list[count % len(ff_list)]
        eq = EquationInfix(ff)
        coeffs = get_all_coeffs(eq.num_coeffs, [-3, 3])
        Y = eq.f(c=coeffs, x=x)
        attempt_count += 1
        if not np.any(np.isnan(Y)):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    normalized_Y = np.around(normalize(Y), 7).tolist()
                    if normalized_Y not in dataset_inputs and normalized_Y not in other_dataset_inputs:
                        dataset_inputs.append(normalized_Y)
                        tokenized_ff = tokenize_eq(ff)
                        dataset_outputs.append(tokenized_ff)
                        eq_with_coeff_list.append(eq.place_exact_coeffs(coeffs))

                        print('.', flush=True, end='')
                        count += 1
                        attempt_count = 0

        if attempt_count > 100:
            del ff_list[count % len(ff_list)]
            attempt_count = 0

    print()
    return dataset_inputs, dataset_outputs, eq_with_coeff_list


def save(dataset_inputs, dataset_outputs, eq_with_coeff_list, save_name):
    save_loc = os.path.join('..', '..', '..', 'datasets')

    filename = 'equations_with_coeff'+save_name+'.csv'
    pd.DataFrame(eq_with_coeff_list).to_csv(os.path.join(save_loc, filename),
                                            index=False, header=None)

    inputs_tensor = torch.Tensor(dataset_inputs)

    # pad dataset_output
    max_len = max([len(out) for out in dataset_outputs])
    for i, out in enumerate(dataset_outputs):
        dataset_outputs[i] = out+[0]*(max_len-len(out))
    dataset_outputs = torch.LongTensor(dataset_outputs)

    dataset = TensorDataset(inputs_tensor, dataset_outputs)
    torch.save(dataset, os.path.join(save_loc, 'dataset'+save_name+'.pt'))


if __name__ == '__main__':
    np.random.seed(0)

    ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
    np.random.shuffle(ff_list)
    ff_list = ff_list[:1000]
    x = np.arange(0.1, 3.1, 0.1)

    # first get 1/4 of the dataset the typical way
    # (randomly choose coefficients for 1/4 of the
    # functional forms)
    dataset_parts = get_dataset(ff_list=ff_list[:250],
                                x=x,
                                dataset_size=12500)
    save(*dataset_parts, save_name='_train_ff1000_confusing_fourth')
