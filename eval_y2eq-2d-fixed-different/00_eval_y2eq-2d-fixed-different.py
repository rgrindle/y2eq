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
from plot2eq.equation.EquationInfix import EquationInfix

from srvgd.utils.normalize import normalize
from srvgd.eval_scripts.order_2d import order_2d
from plot2eq.data_gathering.get_2d_grid import get_2d_grid
from plot2eq.data_gathering.get_2d_points import get_2d_points

import torch
import pandas as pd
import numpy as np

import json

np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000_2d.csv', header=None).values.flatten()

x_int = get_2d_grid(1024)
x_ext = get_2d_grid(1024, a=3.1, b=6.1)

x_list = []
y_normalized_list = []
y_unnormalized_list = []
y_ext_unnormalized_list = []
for eq in eq_list:
    eq = EquationInfix(eq, apply_coeffs=False)
    x_input = get_2d_points(1024)
    x_list.append(x_input.tolist())
    y = eq.f(x_input.T)
    if np.any(np.isnan(y)):
        print('Found eq with nan')
        continue
    points = np.hstack((x_input, y[:, None]))
    y = order_2d(points)
    y_unnormalized_list.append(eq.f(x_int.T).tolist())
    y_normalized_list.append(normalize(y)[:, None].tolist())
    y_ext_unnormalized_list.append(eq.f(x_ext.T).tolist())

with open('00_x_list.json', 'w') as file:
    json.dump(x_list, file)

with open('00_y_normalized_list.json', 'w') as file:
    json.dump(y_normalized_list, file)

with open('00_y_unnormalized_list.json', 'w') as file:
    json.dump(y_unnormalized_list, file)

with open('00_y_ext_unnormalized_list.json', 'w') as file:
    json.dump(y_ext_unnormalized_list, file)
