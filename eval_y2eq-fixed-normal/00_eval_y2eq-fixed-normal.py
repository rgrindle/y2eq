"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Confirm that fixed x values used during training
         is a problem for the current system. Step 0 (this
         script) is to update test dataset with x values
         pulled from a uniform distribution in the same
         interval originally used.

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.data_gathering.get_normal_x import get_normal_x

import torch
import pandas as pd
import numpy as np

import json

np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

x_fixed = np.arange(0.1, 3.1, 0.1)
x_ext = np.arange(3.1, 6.1, 0.1)

x_list = []
y_input_list = []
y_fixed_list = []
y_ext_fixed_list = []
for eq in eq_list:
    eq = EquationInfix(eq, apply_coeffs=False)
    x_int = get_normal_x(num_points=30, mean_radius=0)
    x_int.sort()
    x_list.append(x_int.tolist())
    y = eq.f(x_int)[:, None]
    y_input_list.append(normalize(y).tolist())
    y_fixed_list.append(eq.f(x_fixed).tolist())
    y_ext_fixed_list.append(eq.f(x_ext).tolist())

with open('00_x_list.json', 'w') as file:
    json.dump(x_list, file)

with open('00_y_fixed_list.json', 'w') as file:
    json.dump(y_fixed_list, file)

with open('00_y_input_list.json', 'w') as file:
    json.dump(y_input_list, file)

with open('00_y_ext_fixed_list.json', 'w') as file:
    json.dump(y_ext_fixed_list, file)
