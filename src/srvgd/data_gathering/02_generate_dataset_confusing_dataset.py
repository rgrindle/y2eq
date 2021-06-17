"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 17, 2021

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
import numpy as np
import pandas as pd

import os
import json
import argparse


def add_to_dataset(dataset_inputs, dataset_outputs,
                   eq_with_coeff_list, x, new_ff,
                   other_dataset_inputs=None,
                   max_attempts=1000):
    if other_dataset_inputs is None:
        other_dataset_inputs = []

    eq = EquationInfix(new_ff, x)

    true_y_indices = np.random.choice(len(dataset_inputs), size=max_attempts, replace=False)
    true_y_list = np.array(dataset_inputs)[true_y_indices]

    obs_added_to_dataset = 0
    new_dataset_input_list = []
    new_dataset_output_list = []
    new_eq_with_coeffs_list = []
    for true_y in true_y_list:
        eq.fit(true_y)
        eq.coeffs = np.around(eq.coeffs, 3)
        Y = eq.f(c=eq.coeffs, x=x)
        if not np.any(np.isnan(Y)):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    normalized_Y = np.around(normalize(Y), 7).tolist()
                    if normalized_Y not in dataset_inputs and normalized_Y not in other_dataset_inputs:
                        new_dataset_input_list.append(normalized_Y)
                        new_dataset_output_list.append(tokenize_eq(new_ff))
                        new_eq_with_coeffs_list.append(eq.place_exact_coeffs(eq.coeffs))
                        obs_added_to_dataset += 1
                        print('.', flush=True, end='')
                        if obs_added_to_dataset == 50:
                            break

    print('\nFound', obs_added_to_dataset, 'new observation to add to dataset.')
    return new_dataset_input_list, new_dataset_output_list, new_eq_with_coeffs_list


def save(dataset_inputs, dataset_outputs, eq_with_coeff_list, save_name):
    save_loc = os.path.join('confusing_dataset_in_pieces',
                            save_name+'.json')

    with open(save_loc, 'w') as file:
        json.dump({'eq_list': eq_with_coeff_list,
                   'input': dataset_inputs,
                   'output': dataset_outputs}, file)

    # filename = 'equations_with_coeff'+save_name+'.csv'
    # pd.DataFrame(eq_with_coeff_list).to_csv(os.path.join(save_loc, filename),
    #                                         index=False, header=None)

    # inputs_tensor = torch.Tensor(dataset_inputs)

    # # pad dataset_output
    # max_len = max([len(out) for out in dataset_outputs])
    # for i, out in enumerate(dataset_outputs):
    #     dataset_outputs[i] = out+[0]*(max_len-len(out))
    # dataset_outputs = torch.LongTensor(dataset_outputs)

    # dataset = TensorDataset(inputs_tensor, dataset_outputs)
    # torch.save(dataset, os.path.join(save_loc, 'dataset'+save_name+'.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_num', type=int,
                        required=True,
                        help='This is the new ff index. '
                             'the index used to pick the ff '
                             'that will be fit to an existing instance '
                             'of a ff in the fourth of the dataset.')

    args = parser.parse_args()

    os.makedirs('confusing_dataset_in_pieces', exist_ok=True)

    # Need this seed so that shuffle
    # will be same as script 01.
    np.random.seed(0)

    # Get all the ff's
    ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
    np.random.shuffle(ff_list)
    ff_list = ff_list[:1000]
    x = np.arange(0.1, 3.1, 0.1)

    # I want to make sure we get a different order
    # to attempt to fit the new ff into one of
    # the existing instances.
    np.random.seed(args.job_num)

    # Get the first 1/4 of the dataset that was
    # created by previous script
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000_confusing_fourth.pt'))
    dataset_inputs = [d[0].tolist() for d in dataset]
    dataset_outputs = [d[1].tolist() for d in dataset]
    eq_with_coeff_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000_confusing_fourth.csv'), header=None)
    print(len(dataset_inputs), len(dataset_outputs), len(eq_with_coeff_list))

    dataset_parts = add_to_dataset(dataset_inputs=dataset_inputs,
                                   dataset_outputs=dataset_outputs,
                                   eq_with_coeff_list=eq_with_coeff_list,
                                   new_ff=ff_list[250+args.job_num],
                                   x=x)

    save(*dataset_parts, save_name='_train_ff1000_confusing_jobnum{}'.format(args.job_num))
