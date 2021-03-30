"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 2, 2021

PURPOSE: Get RMSE of functional forms saved in get_ff.
         First coefficients are determined with non-linear
         regression algorithm.

NOTES: Empty strings as "equations" may happen if there are
       no math tokens in the "equation" (e.g. STARTEND, or STARTPADPAD...)
       pandas interprets these a nan.

TODO:
"""
from srvgd.utils.eval import regression, normalize, get_f, apply_coeffs, RMSE, is_eq_valid

import pandas as pd
import numpy as np


def get_rmse(x_int_list, y_int_list, x_ext, y_ext_list, ff_list):
    """
    Fit coefficients using y_int and get rmse for
    both y_int and y_ext.
    """

    rmse_int_list = []
    rmse_ext_list = []
    for i, (ff, y_int, x_int, y_ext) in enumerate(zip(ff_list, y_int_list, x_int_list, y_ext_list)):
        if pd.isnull(ff):
            rmse_int = float('inf')
            rmse_ext = float('inf')
        elif is_eq_valid(ff):
            ff_coeff, num_coeffs = apply_coeffs(ff)
            f_hat = get_f(ff_coeff)
            coeffs, rmse_int = regression(f_hat, y_int, num_coeffs, np.array(x_int))

            y_int_true_norm, true_min_, true_scale = normalize(y_int, return_params=True)
            y_int_pred_norm = normalize(f_hat(c=coeffs, x=np.array(x_int)), true_min_, true_scale)

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