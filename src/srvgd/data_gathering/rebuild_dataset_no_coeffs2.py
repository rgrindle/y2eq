"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 24, 2021

PURPOSE: Make a version of dataset where all coefficients
         are always 1.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize

import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

import os

full_ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
test_ff_list = pd.read_csv('../../../datasets/test_ff_list.csv', header=None).values.flatten()
print(len(full_ff_list), len(test_ff_list))
train_ff_list = [eq for eq in full_ff_list if eq not in test_ff_list]
print(len(train_ff_list))

eq_list = [EquationInfix(ff, apply_coeffs=False) for ff in train_ff_list]

x = np.arange(0.1, 3.1, 0.1)
dataset_input = [normalize(eq.f(x))[:, None].tolist() for eq in eq_list]

dataset_output = [tokenize_eq(ff) for ff in train_ff_list]

# pad dataset_output
max_len = max([len(out) for out in dataset_output])
padded_dataset_output = []
for i, out in enumerate(dataset_output):
    dataset_output[i] = out+[0]*(max_len-len(out))

print(len(dataset_input))
print(len(dataset_output))

dataset_path = os.path.join('..', '..', '..', 'datasets')
dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_no_coeffs2.pt'))
