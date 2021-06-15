"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 4, 2021

PURPOSE: Compute root mean squared error of functional forms
         output by neural network

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.utils.rmse import RMSE

import pandas as pd
import numpy as np

import json

with open('00_y_int_fixed_unnormalized_list.json', 'r') as json_file:
    y_int_fixed_unnormalized_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_unnormalized_list = json.load(json_file)

ff_list = pd.read_csv('01_predicted_ff.csv', header=None).values.flatten()


def get_rmse(y_int_fixed_list, y_ext_list, ff_list):
    x_int = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    rmse_int_list = []
    rmse_ext_list = []
    for i, (ff, y_int_fixed, y_ext) in enumerate(zip(ff_list, y_int_fixed_list, y_ext_list)):
        y_int = np.array(y_int_fixed).flatten()

        if pd.isnull(ff):
            rmse_int = np.inf
            rmse_ext = np.inf

        else:
            eq = EquationInfix(ff, x=x_int, apply_coeffs=False)

            if eq.is_valid():
                y_int_true_norm, true_min_, true_scale = normalize(y_int, return_params=True)
                y_int_pred_norm = normalize(eq.f(x=x_int).flatten(), true_min_, true_scale)

                y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
                y_ext_pred_norm = normalize(eq.f(x=x_ext).flatten(), true_min_, true_scale)

                rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)
                rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)

            else:
                rmse_int = np.inf
                rmse_ext = np.inf

        rmse_int_list.append(rmse_int)
        rmse_ext_list.append(rmse_ext)
        print(i, rmse_int_list[-1], rmse_ext_list[-1])

    pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('02_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])


get_rmse(y_int_fixed_list=y_int_fixed_unnormalized_list,
         y_ext_list=y_ext_unnormalized_list,
         ff_list=ff_list)
