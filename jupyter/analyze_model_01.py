"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 18, 2021

PURPOSE: Count the number of valid and number of invalid equations.

NOTES: Requires test_output.csv containing rows of numbers
       that can be decoded to a string/equation using
       eqlearner.dataset.processing.tokenization.get_string()

TODO: Get RMSE values here too.
"""
from srvgd.utils.eval import is_eq_valid
from eqlearner.dataset.processing.tokenization import get_string

import numpy as np
import pandas as pd
import tokenize
import sympy
from sympy import sin, exp, log, lambdify, Symbol

import json

np.seterr('raise')

file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
# file_endname = '_epochs100_0'
data = pd.read_csv('test_output{}.csv'.format(file_endname)).values
print(data.shape)

# decode and get y-values
valid = 0
invalid = 0
y_hat_values = {}
valid_equations = {}
for i, d in enumerate(data):
    eq_str = get_string(d)
    end = eq_str.find('END')
    if end != -1:
        eq_str = eq_str[:end]
    if is_eq_valid(eq_str):
        valid_equations[i] = eq_str
        valid += 1
    else:
        invalid += 1


print('valid', valid)
print('invalid', invalid)

with open('01_valid_eq{}.json'.format(file_endname), 'w') as file:
    json.dump(valid_equations, file)
