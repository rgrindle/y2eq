"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 4, 2021

PURPOSE: Compute root mean squared error of functional forms
         output by neural network but fix ones that have an
         extra parenthesis at the end.

NOTES:

TODO:
"""
from srvgd.utils.attempt_to_make_valid import attempt_to_make_valid

from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.utils.rmse import RMSE

import numpy as np
import pandas as pd

import json

with open('00_y_int_fixed_unnormalized_list.json', 'r') as json_file:
    y_int_fixed_unnormalized_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_unnormalized_list = json.load(json_file)

eq_list = pd.read_csv('01_predicted_ff.csv', header=None).values.flatten()

# The following is modified version of
# fit_coeffs_and_get_rmse from from srvgd.utils.eval
# This version does not fit the functional form because
# the NN has output a full equation.
x_int = np.arange(0.1, 3.1, 0.1)
x_ext = np.arange(3.1, 6.1, 0.1)

rmse_int_list = []
rmse_ext_list = []
invalid_eq_list = []
for i, (eq_str, y_int_fixed, y_ext) in enumerate(zip(eq_list, y_int_fixed_unnormalized_list, y_ext_unnormalized_list)):
    y_int = np.array(y_int_fixed).flatten()

    if pd.isnull(eq_str):
        rmse_int = np.inf
        rmse_ext = np.inf

    else:
        eq = attempt_to_make_valid(eq_str, x_int)

        if eq is None:
            rmse_int = np.inf
            rmse_ext = np.inf

        else:

            y_int_true_norm, true_min_, true_scale = normalize(y_int, return_params=True)
            y_int_pred_norm = normalize(eq.f(x=x_int).flatten(), true_min_, true_scale)

            y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
            y_ext_pred_norm = normalize(eq.f(x=x_ext).flatten(), true_min_, true_scale)

            rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)
            rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)

    rmse_int_list.append(rmse_int)
    rmse_ext_list.append(rmse_ext)
    print(i, rmse_int_list[-1], rmse_ext_list[-1])

pd.DataFrame(invalid_eq_list).to_csv('invalid_eq_list.csv', index=False, header=None)
pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('04_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])
