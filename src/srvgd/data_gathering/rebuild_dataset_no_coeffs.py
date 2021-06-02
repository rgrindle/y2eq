"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 24, 2021

PURPOSE: Make a version of dataset where all coefficients
         are always 1.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize

import torch
from torch.utils.data import TensorDataset
import numpy as np

import os


dataset_path = os.path.join('..', '..', '..', 'datasets')

dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                     map_location=torch.device('cpu'))

dataset_output = [d[1].tolist() for d in dataset]
ff_list = [get_eq_string(out)[5:-3] for out in dataset_output]
eq_list = [EquationInfix(ff, apply_coeffs=False) for ff in ff_list]

x = np.arange(0.1, 3.1, 0.1)
dataset_input = [normalize(eq.f(x))[:, None].tolist() for eq in eq_list]

# pad dataset_output
max_len = max([len(out) for out in dataset_output])
padded_dataset_output = []
for i, out in enumerate(dataset_output):
    dataset_output[i] = out+[0]*(max_len-len(out))

print(len(dataset_input))
print(len(dataset_output))

dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_no_coeffs.pt'))
