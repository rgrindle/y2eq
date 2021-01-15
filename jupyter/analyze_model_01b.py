"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 15, 2021

PURPOSE: Count the number of valid and number of invalid equations.
         But, do it without considering STOP/END. Instead shorten
         the equation until it is valid. Also save the valid equations
         with index for RMSE computation later.

NOTES: Requires test_output.csv containing rows of numbers
       that can be decoded to a string/equation using
       eqlearner.dataset.processing.tokenization.get_string()

TODO:
"""
from eqlearner.dataset.processing.tokenization import default_map, reverse_map

import json
import numpy as np
import pandas as pd
import tokenize
import sympy
from sympy import sin, exp, log, lambdify, Symbol

np.seterr('raise')

file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
data = pd.read_csv('test_output{}.csv'.format(file_endname)).values
print(data.shape)


def get_string(string, mapping=None, sym_mapping=None):
    if not mapping:
        tmp = default_map()
        mapping = reverse_map(tmp, symbols=sym_mapping)
    mapping_string = mapping.copy()
    mapping_string[12] = "START"
    mapping_string[13] = "END"
    mapping_string[0] = ''
    curr = "".join([mapping_string[digit] for digit in string])
    # if len(string) < 2:
    #     return RuntimeError
    # if len(string) == 2:
    #     return 0
    return curr


def is_eq_valid(eq_str):
    try:
        f = lambdify(x, eq_str)
        y_hat_values = f(x_numeric)
        return type(y_hat_values) != np.ufunc
    except (sympy.SympifyError, TypeError, NameError, tokenize.TokenError, AttributeError, FloatingPointError) as e:
        if str(e) == AttributeError:
            print('AttributeError on', eq_str)
        return False


# decode and get y-values
valid = 0
invalid = 0
valid_equations = {}
x_numeric = np.arange(0.1, 3.1, 0.1)
x = Symbol('x', real=True)
for i, d in enumerate(data):
    for end in range(len(d), 0, -1):
        eq_str = str(get_string(d[:end])).replace('END', '')
        if is_eq_valid(eq_str):
            print(eq_str)
            valid_equations[i] = eq_str
            valid += 1
            break
    else:
        invalid += 1
    print(valid, invalid)

print('valid', valid)
print('invalid', invalid)

with open('valid_eq{}.json'.format(file_endname), 'w') as file:
    json.dump(valid_equations, file)
