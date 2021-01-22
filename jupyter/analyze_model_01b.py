"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 21, 2021

PURPOSE: Count the number of valid and number of invalid equations.
         But, do it without considering STOP/END. Instead shorten
         the equation until it is valid. Also save the valid equations
         with index for RMSE computation later.

NOTES: Requires test_output.csv containing rows of numbers
       that can be decoded to a string/equation using
       eqlearner.dataset.processing.tokenization.get_string()

TODO:
"""
from srvgd.utils.eval import is_eq_valid
from eqlearner.dataset.processing.tokenization import default_map, reverse_map

import json
import numpy as np
import pandas as pd
import tokenize
import sympy
from sympy import sin, exp, log, lambdify, Symbol

np.seterr('raise')

# file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
file_endname = '_epochs100_0'
data = pd.read_csv('test_output{}.csv'.format(file_endname)).values.flatten()
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


# decode and get y-values
valid = 0
invalid = 0
valid_equations = {}
x_numeric = np.arange(0.1, 3.1, 0.1)
x = Symbol('x', real=True)
for i, eq_str in enumerate(data):
    for end in range(len(eq_str), 0, -1):
        eq_str = eq_str[:end].replace('END', '').replace('START', '')
        if is_eq_valid(eq_str):
            # print(eq_str)
            valid_equations[i] = eq_str
            valid += 1
            break
    else:
        invalid += 1
        print(eq_str)
    print(valid, invalid)

print('valid', valid)
print('invalid', invalid)

with open('01b_valid_eq{}.json'.format(file_endname), 'w') as file:
    json.dump(valid_equations, file)
