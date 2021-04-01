"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Confirm that fixed x values used during training
         is a problem for the current system. Step 2 (this
         script) is to compute the RMSE of the output equations
         from step one and the y-values from step 0.

NOTES: Empty strings as "equations" may happen if there are
       no math tokens in the "equation" (e.g. STARTEND, or STARTPADPAD...)
       pandas interprets these a nan.

TODO:
"""
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import RMSE
from plot2eq.equation.EquationInfix import EquationInfix
from plot2eq.data_gathering.get_2d_grid import get_2d_grid

import pandas as pd
import numpy as np

import json

with open('00_x_list.json', 'r') as json_file:
    x_list = json.load(json_file)

with open('00_y_unnormalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_list = json.load(json_file)

ff_list = pd.read_csv('01_predicted_ff.csv', header=None).values.flatten()

x_int = get_2d_grid(1024, a=0.1, b=3.1)
x_ext = get_2d_grid(1024, a=3.1, b=6.1)

rmse_int_list = []
rmse_ext_list = []
for i, (ff, y, x, y_ext) in enumerate(zip(ff_list, y_list, x_list, y_ext_list)):

    if pd.isnull(ff):
        rmse_int = float('inf')
        rmse_ext = float('inf')
    else:
        ff = ff.replace('x0', 'x[0]').replace('x1', 'x[1]')
        try:
            eq = EquationInfix(ff, x=np.array(x_int).T)
            valid = True
        except SyntaxError:
            valid = False
            int_rmse = float('inf')
            ext_rmse = float('inf')

        if valid:
            eq.fit(y)

            y_int_true_norm, true_min_, true_scale = normalize(y, return_params=True)
            y_int_pred_norm = normalize(eq.f(c=eq.coeffs, x=np.array(x).T), true_min_, true_scale)

            y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
            y_ext_pred_norm = normalize(eq.f(c=eq.coeffs, x=x_ext.T), true_min_, true_scale)

            rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)
            rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)

    rmse_ext_list.append(rmse_ext)
    rmse_int_list.append(rmse_int)
    print(i, rmse_int_list[-1], rmse_ext_list[-1])

pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('02_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])
