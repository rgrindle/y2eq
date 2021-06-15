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

NOTES: In 02 (this script), use the fourth of the dataset
       created by 01 as a starting point. With the remaining
       functional forms create more observations that have
       similar y-values to the original fourth of the dataset.

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
import copy


def add_to_dataset(_dataset_inputs, _dataset_outputs,
                   _eq_with_coeff_list, x, remaining_ff,
                   dataset_size,
                   other_dataset_inputs=None,
                   max_attempts=100):
    if other_dataset_inputs is None:
        other_dataset_inputs = []

    dataset_inputs = copy.deepcopy(_dataset_inputs)
    dataset_outputs = copy.deepcopy(_dataset_outputs)
    eq_with_coeff_list = copy.deepcopy(_eq_with_coeff_list)

    count = 0
    attempt_count = 0
    while count < dataset_size:
        ff = remaining_ff[count % len(ff_list)]
        eq = EquationInfix(ff, x)

        if attempt_count == 0:
            true_y_list = np.random.choice(_dataset_inputs, size=max_attempts, replace=False)

        true_y = true_y_list[attempt_count]
        eq.fit(true_y)
        Y = eq.f(c=eq.coeffs, x=x)
        attempt_count += 1
        if not np.any(np.isnan(Y)):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    normalized_Y = np.around(normalize(Y), 7).tolist()
                    if normalized_Y not in dataset_inputs and normalized_Y not in other_dataset_inputs:
                        dataset_inputs.append(normalized_Y)
                        tokenized_ff = tokenize_eq(ff)
                        dataset_outputs.append(torch.Tensor(tokenized_ff))
                        eq_with_coeff_list.append(eq.place_exact_coeffs(eq.coeffs))

                        print('.', flush=True, end='')
                        count += 1
                        attempt_count = 0
                        exit()

        if attempt_count > max_attempts:
            del ff_list[count % len(ff_list)]
            attempt_count = 0

    print()
    return dataset_inputs, dataset_outputs, eq_with_coeff_list


def save(dataset_inputs, dataset_outputs, eq_with_coeff_list, save_name):
    save_loc = os.path.join('..', '..', '..', 'datasets')

    inputs_tensor = torch.zeros(len(dataset_inputs), 30)
    for i, y in enumerate(dataset_inputs):
        inputs_tensor[i, :] = torch.Tensor(y)

    filename = 'equations_with_coeff'+save_name+'.csv'
    pd.DataFrame(eq_with_coeff_list).to_csv(os.path.join(save_loc, filename),
                                            index=False, header=None)

    dataset = TensorDataset(inputs_tensor, pad(dataset_outputs))
    torch.save(dataset, os.path.join(save_loc, 'dataset'+save_name+'.pt'))


if __name__ == '__main__':
    np.random.seed(0)

    ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
    np.random.shuffle(ff_list)
    ff_list = ff_list[:1000]
    x = np.arange(0.1, 3.1, 0.1)

    # Get the first 1/4 of the dataset that was
    # created by previous script
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000_confusing_fourth.pt'))
    dataset_inputs = [d[0].tolist() for d in dataset]
    dataset_outputs = [d[1].tolist() for d in dataset]
    eq_with_coeff_list = pd.read_csv(os.path.join(dataset_path, 'eq_with_coeff_ff1000_confusing_fourth.csv'))
    dataset_parts = (dataset_inputs, dataset_outputs, eq_with_coeff_list)
    exit()

    dataset_parts = add_to_dataset(*dataset_parts,
                                   remaining_ff=ff_list[250:],
                                   x=x)
    save(*dataset_parts, save_name='_train_ff1000_confusing')
