"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Fit the 80 unique output ff's to the 1000 functional
         forms and compute the RMSE.

NOTES:

TODO:
"""
from srvgd.utils.eval import regression, normalize, get_f, apply_coeffs, RMSE, is_eq_valid
import pandas as pd
import numpy as np

import json

with open('../../../eval_y2eq-fixed-fixed/00_y_normalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)

output_ff_list = pd.read_csv('unique_output_ff_list.csv', header=None).values.flatten()

x_int = np.arange(0.1, 3.1, 0.1)

rmse_int_dict = {}
for k, ff in enumerate(output_ff_list):
    rmse_int_dict[ff] = []
    for i, y in enumerate(y_list):
        rmse_int = float('inf')
        if not pd.isnull(ff) and is_eq_valid(ff):
            ff_coeff, num_coeffs = apply_coeffs(ff)
            f_hat = get_f(ff_coeff)
            coeffs, rmse_int = regression(f_hat, y, num_coeffs, np.array(x_int))

            y_int_true_norm, true_min_, true_scale = normalize(y, return_params=True)
            y_int_pred_norm = normalize(f_hat(c=coeffs, x=np.array(x_int)), true_min_, true_scale)

            rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)

        rmse_int_dict[ff].append(rmse_int)
        print(k, i, rmse_int_dict[ff][-1])

pd.DataFrame(rmse_int_dict.values()).T.to_csv('rmse.csv', index=False, header=list(rmse_int_dict.keys()))
