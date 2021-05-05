"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Figure out in what way invalid equations output by
         the NN are invalid. If possible, fix them.

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix

import numpy as np
import pandas as pd

invalid_eq_list = pd.read_csv('invalid_eq_list.csv', header=None).values.flatten()

x_int = np.arange(0.1, 3.1, 0.1)

valid_count = 0
for eq in invalid_eq_list:
    eq = EquationInfix(eq, x=x_int, apply_coeffs=False)

    if eq.is_valid():
        print('ERROR: THIS ONE IS VALID?: ', eq.eq_str)
        exit()

    elif eq.eq_str[-1] != ')':
        print('Does not end in ):', eq.eq_str)

    else:
        eq = EquationInfix(eq.eq_str[:-1], x=x_int, apply_coeffs=False)
        if eq.is_valid():
            valid_count += 1
        else:
            print('Invalid after fix:', eq.eq_str)

print('valid_count', valid_count, 'of', len(invalid_eq_list))
