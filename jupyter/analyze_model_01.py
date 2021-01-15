"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 5, 2021

PURPOSE: Count the number of valid and number of invalid equations.

NOTES: Requires test_output.csv containing rows of numbers
       that can be decoded to a string/equation using
       eqlearner.dataset.processing.tokenization.get_string()

TODO: Get RMSE values here too.
"""
from eqlearner.dataset.processing.tokenization import get_string

import numpy as np
import pandas as pd
import tokenize
import sympy
from sympy import sin, exp, log, lambdify, Symbol

np.seterr('raise')

file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates'
# file_endname = '_epochs100_0'
data = pd.read_csv('test_output{}.csv'.format(file_endname)).values
print(data.shape)

# decode and get y-values
valid = 0
invalid = 0
y_hat_values = {}
x_numeric = np.arange(0.1, 3.1, 0.1)
x = Symbol('x', real=True)
for i, d in enumerate(data):
    eq_str = get_string(d)
    end = eq_str.find('END')
    if end != -1:
        eq_str = eq_str[:end]
    try:
        f = lambdify(x, eq_str)
        y_hat_values[i] = f(x_numeric)
        if type(y_hat_values[i]) == np.ufunc:
            del y_hat_values[i]
            invalid += 1
            print(eq_str)
        else:
            valid += 1
    except (sympy.SympifyError, TypeError, NameError, tokenize.TokenError, FloatingPointError) as e:
        invalid += 1
        print(eq_str)

print('valid', valid)
print('invalid', invalid)

# # get RMSE
# import torch
# from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

# rmse_list = []
# test_data = torch.load('train_data_int_comp.pt', map_location=torch.device('cpu'))
# y_values = [d[0].numpy() for d in test_data]
# for key in y_hat_values:
#     rmse = np.sqrt(np.mean(np.power(y_values[key]-y_hat_values[key], 2)))
#     rmse_list.append(rmse)

# pd.DataFrame(rmse_list).to_csv('rmse_test.csv')
