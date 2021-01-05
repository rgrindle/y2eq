"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 5, 2020

PURPOSE: Count the number of valid and number of invalid equations.
         But, do it without considering STOP/END. Instead shorten
         the equation until it is valid.

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

data = pd.read_csv('test_output.csv').values
print(data.shape)


def is_eq_valid(eq_str):
    try:
        f = lambdify(x, eq_str)
        y_hat_values[i] = f(x_numeric)
        if type(y_hat_values[i]) == np.ufunc:
            del y_hat_values[i]
            return False
        else:
            return True
    except (sympy.SympifyError, TypeError, NameError, tokenize.TokenError, FloatingPointError) as e:
        return False


# decode and get y-values
valid = 0
invalid = 0
y_hat_values = {}
x_numeric = np.arange(0.1, 3.1, 0.1)
x = Symbol('x', real=True)
for i, d in enumerate(data):
    eq_str = get_string(d).replace('END', '')
    for end in range(len(eq_str), 0, -1):
        # print(eq_str[:end])
        if is_eq_valid(eq_str[:end]):
            print(eq_str[:end])
            valid += 1
            break
    else:
        invalid += 1
    print(valid, invalid)

print('valid', valid)
print('invalid', invalid)
