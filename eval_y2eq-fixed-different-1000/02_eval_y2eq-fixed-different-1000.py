"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 27, 2021

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
from srvgd.utils.eval import regression, get_f, apply_coeffs, RMSE, is_eq_valid

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

x_ext = np.arange(3.1, 6.1, 0.1)

rmse_int_list = []
rmse_ext_list = []
for i, (ff, y, x, y_ext) in enumerate(zip(ff_list, y_list, x_list, y_ext_list)):
    if pd.isnull(ff):
        rmse_int = float('inf')
        rmse_ext = float('inf')
    elif is_eq_valid(ff):
        ff_coeff, num_coeffs = apply_coeffs(ff)
        f_hat = get_f(ff_coeff)
        coeffs, rmse_int = regression(f_hat, y, num_coeffs, np.array(x))

        y_int_true_norm, true_min_, true_scale = normalize(y, return_params=True)
        y_int_pred_norm = normalize(f_hat(c=coeffs, x=np.array(x)), true_min_, true_scale)

        y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
        y_ext_pred_norm = normalize(f_hat(c=coeffs, x=x_ext), true_min_, true_scale)

        rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)
        rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)
    else:
        rmse_int = float('inf')
        rmse_ext = float('inf')

    rmse_ext_list.append(rmse_ext)
    rmse_int_list.append(rmse_int)
    print(i, rmse_int_list[-1], rmse_ext_list[-1])

pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('02_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])
