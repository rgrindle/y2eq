"""
AUTHOR: Ryan Grindle

LAST MODIFIED: April 2, 2021

PURPOSE: Evaluate non-linear optimization algorithm
         on regression of functional forms from the
         dataset.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import get_f

import torch
import pandas as pd
import numpy as np

import json

np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

dataset = torch.load('../datasets/dataset_test_ff1000.pt', map_location=device)
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]

x_int = np.arange(0.1, 3.1, 0.1)
x_ext = np.arange(3.1, 6.1, 0.1)

x_list = []
y_normalized_list = []
y_unnormalized_list = []
y_ext_unnormalized_list = []
new_ff_list = []
for i, eq in enumerate(eq_list):

    f = get_f(eq)
    x_list.append(x_int.tolist())
    y = f(x_int)

    if np.any(np.isnan(y)):
        print('found nan')
        exit()

    y_unnormalized_list.append(y.tolist())
    y_normalized_list.append(normalize(y)[:, None].tolist())
    y_ext_unnormalized_list.append(f(x_ext).tolist())
    new_ff_list.append(ff_list[i])

print(len(x_list))
assert len(x_list) == len(y_normalized_list)
assert len(x_list) == len(y_unnormalized_list)
assert len(x_list) == len(y_ext_unnormalized_list)
assert len(x_list) == len(new_ff_list)

with open('00_x_list.json', 'w') as file:
    json.dump(x_list, file)

with open('00_y_normalized_list.json', 'w') as file:
    json.dump(y_normalized_list, file)

with open('00_y_unnormalized_list.json', 'w') as file:
    json.dump(y_unnormalized_list, file)

with open('00_y_ext_unnormalized_list.json', 'w') as file:
    json.dump(y_ext_unnormalized_list, file)

pd.DataFrame(new_ff_list).to_csv('00_ff_list.csv', index=False, header=None)
