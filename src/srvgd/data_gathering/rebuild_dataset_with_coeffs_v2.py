"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Make a new version of dataset_train_ff1000.pt by
         replacing the functional forms with the equations
         with the coefficients specified. This dataset could
         be used to train a version of y2eq (or plot2eq) that
         does not need an optimization algorithm to fit the
         coefficients.

NOTES: This version will store the coefficient values separately
       so that values get a MSE loss.

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import numberize_tokens, extract_tokens

import torch
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

import os

dataset_path = os.path.join('..', '..', '..', 'datasets')

eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000.csv'),
                      header=None).values.flatten()

dataset_output_cat = []
dataset_output_num = []
for eq in eq_list:
    tokenized_eq = extract_tokens(eq, group_minus_signs=True)

    # Numbers are still grouped (e.g. 1.234 is one token).
    # Make each digit (and decimal point) a separate token.
    new_tokenized_eq = []
    coeff_value_list = []
    for token in tokenized_eq:
        if '.' in token:
            coeff_value_list.append(float(token))
            new_tokenized_eq.extend('c')
        else:
            coeff_value_list.append(float('NaN'))
            new_tokenized_eq.append(token)
    numberized_eq = numberize_tokens(new_tokenized_eq, two_d=False, include_coeffs=True)
    dataset_output_cat.append(numberized_eq)
    dataset_output_num.append(coeff_value_list)


# pad dataset_output
max_len = max([len(out) for out in dataset_output_cat])
padded_dataset_output = []
for i, out in enumerate(dataset_output_cat):
    dataset_output_cat[i] = out+[0]*(max_len-len(out))
    dataset_output_num[i] = dataset_output_num[i]+[float('NaN')]*(max_len-len(out))

# Get y-values from existing dataset
dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                     map_location=torch.device('cpu'))
dataset_input = [d[0].tolist() for d in dataset]

print(len(dataset_input))
print(len(dataset_output_cat), len(dataset_output_num))

dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output_cat), torch.Tensor(dataset_output_num))
torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_with_coeffs_v2.pt'))
