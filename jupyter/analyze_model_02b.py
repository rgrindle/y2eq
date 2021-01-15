"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 15, 2021

PURPOSE: Compute coefficients for valid equations found in
         analyze_model_01b.py

NOTES:

TODO:
"""
from srvgd.utils.eval import fit_eq
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import json
import torch
import numpy as np
import pandas as pd
import tokenize
import sympy
from sympy import sin, exp, log, lambdify, Symbol


# Get valid equations
file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
with open('valid_eq{}.json'.format(file_endname), 'r') as json_file:
    valid_equations = json.load(json_file)

for key in valid_equations:
    if 'x' not in valid_equations[key]:
        valid_equations[key] += '+0*x'

# Get expected y-values
test_data = torch.load('test_data_int_comp.pt', map_location=torch.device('cpu'))
y_true = np.array([d[0].tolist() for d in test_data])
print(y_true.shape)

# eq_data = pd.read_csv('equations_with_coeff_test.csv', header=None).values
# print(eq_data.shape)
# eq_data = [eq_data[int(i)][0] for i in valid_equations]
# for key in valid_equations:
#     print(valid_equations[key])
#     print(eq_data[int(key)])
#     print()

y_list = [y_true[int(i)] for i in valid_equations]
coeff_list, rmse_list, f_list = fit_eq(eq_list=valid_equations.values(),
                                       support=np.arange(0.1, 3.1, 0.1)[:, None],
                                       y_list=y_list)

pd.DataFrame(rmse_list).to_csv('rmse{}.csv'.format(file_endname, index=False, header=None))
