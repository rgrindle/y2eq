"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Figure out in what way invalid equations output by
         the NN are invalid. If possible, fix them.

NOTES:

TODO:
"""
from srvgd.utils.attempt_to_make_valid import attempt_to_make_valid, counts

import numpy as np
import pandas as pd


invalid_eq_list = pd.read_csv('invalid_eq_list.csv', header=None).values.flatten()

x_int = np.arange(0.1, 3.1, 0.1)

valid_count = 0
for eq_str in invalid_eq_list:
    eq = attempt_to_make_valid(eq_str, x_int)

print('total num eq =', len(invalid_eq_list))
print(counts)
