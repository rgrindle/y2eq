"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 18, 2021

PURPOSE: Make the dataset used to train MSR in a form
         that can be used to train y2eq.

NOTES:

TODO:
"""
import PredictingUnseenFunctions.data_gathering.io_dataset as io_dataset
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq, get_eq_string
from srvgd.utils.normalize import normalize

import sympy
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

import os

dataset_file = os.path.join(os.environ['MSR_DATASET_PATH'], 'semantically_unique_dataset_head5.csv')
kexpression_list, x_list, y_list = io_dataset.read_dataset(dataset_file)
# dataset_list = io_dataset.prep_regression_data(x_list, y_list)

print(len(kexpression_list), len(x_list), len(y_list))

# normalize y
normalized_y_list = []
normalization_params_list = []
for i, yi in enumerate(y_list):
    if max(yi) == min(yi):
        print('Skipping constant y-values. index =', i)
        continue

    normalized_y, min_, scale_ = normalize(yi, return_params=True)
    normalized_y_list.append(normalized_y)
    normalization_params_list.append([min_, scale_])

dataset_input = normalized_y_list

eq_list = []
dataset_output = []
for i, kexpr in enumerate(kexpression_list):
    layers = kexpr.get_tree_layers()
    subtree = kexpr.get_subtree_infix(layers)
    eq = ''.join(subtree).replace('x0', 'x')
    eq_sympy = sympy.sympify(eq)
    if str(eq_sympy) == '0':
        print('Skipping 0. index =', i)
        continue
    eq_list.append(str(eq_sympy))
    dataset_output.append(tokenize_eq(eq_list[-1]))
    print(eq_list[-1])
    print(dataset_output[-1])
    print(get_eq_string(dataset_output[-1]))
    print()

# do padding
max_len = max([len(eq) for eq in dataset_output])
for i, eq in enumerate(dataset_output):
    dataset_output[i] = eq+[0]*(max_len-len(eq))

for eq in dataset_output:
    assert len(eq) == len(dataset_output[0])

dataset_output = dataset_output

dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
save_loc = os.path.join('..', '..', '..', 'datasets')
torch.save(dataset, os.path.join(save_loc, 'dataset_msr.pt'))

pd.DataFrame(eq_list).to_csv(os.path.join(save_loc, 'equations_with_coeffs_msr.csv'),
                             header=False)

pd.DataFrame(normalization_params_list).to_csv(os.path.join(save_loc, 'normalization_params_msr.csv'),
                                               header=False, index=False)

# save split dataset.

# get same split as done by MSR
path = '/Users/rgrindle/TlcsrSimplified_data/experiment39'
df_indices = pd.read_csv(os.path.join(path, 'rep9_dataset_indices.csv'),
                         header=None)

indices = {}

for dataset_type in ['train', 'validation', 'test']:
    indices_with_nans = df_indices.loc[df_indices[0] == dataset_type].values[0, 1:].astype(np.float64)
    not_nans = np.logical_not(np.isnan(indices_with_nans))
    indices[dataset_type] = indices_with_nans[not_nans].astype(int).tolist()

# index 5 was the zero function which has been removed,
# so adjust
for key in indices:
    for i, index in enumerate(indices[key]):
        if index < 5:
            pass
        elif index > 5:
            indices[key][i] = index - 1
        else:
            to_remove = (key, i)

indices[to_remove[0]] = [i for i in indices[to_remove[0]] if i != 5]

for key in indices:
    print(len(indices[key]))
    assert 5 not in indices[key]

for key in indices:
    sub_dataset_input = [dataset_input[i] for i in indices[key]]
    sub_dataset_output = [dataset_output[i] for i in indices[key]]
    sub_dataset = TensorDataset(torch.Tensor(sub_dataset_input), torch.LongTensor(sub_dataset_output))
    torch.save(sub_dataset, os.path.join(save_loc, 'dataset_msr_{}.pt'.format(key)))
