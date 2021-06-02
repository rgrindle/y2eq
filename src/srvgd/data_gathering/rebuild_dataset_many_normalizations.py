"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 24, 2021

PURPOSE: Read in dataset_train_ff1000 and multiple instances
         of normalization of y-values.

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


dataset_path = os.path.join('..', '..', '..', 'datasets')

x = np.arange(0.1, 3.1, 0.1)
eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000.csv'), header=None).values.flatten()
eq_list = [EquationInfix(eq, apply_coeffs=False) for eq in eq_list]
unnormalized_y_list = [eq.f(x) for eq in eq_list]

dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                     map_location=torch.device('cpu'))

dataset_output = [d[1].tolist() for d in dataset]

y_normalized1_list = [normalize(y) for y in unnormalized_y_list]
y_normalized2_list, min_, scale_ = normalize(unnormalized_y_list, return_params=True)

# save min_, scale_
pd.DataFrame([min_, scale_]).to_csv(os.path.join(dataset_path, 'normalization_params_many_normalizations.csv'),
                                    index=False, header=None)

dataset_input = np.stack((y_normalized1_list, y_normalized2_list), axis=-1)

# pad dataset_output
max_len = max([len(out) for out in dataset_output])
padded_dataset_output = []
for i, out in enumerate(dataset_output):
    dataset_output[i] = out+[0]*(max_len-len(out))

print(len(dataset_input))
print(len(dataset_output))

dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_many_normalizations.pt'))
