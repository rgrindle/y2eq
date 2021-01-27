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
from srvgd.utils.eval import regression, get_f, apply_coeffs
import pandas as pd
import numpy as np

import json

with open('00_x_list.json', 'r') as json_file:
    x_list = json.load(json_file)

with open('00_y_unnormalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)

ff_list = pd.read_csv('01_predicted_ff.csv', header=None).values.flatten()

rmse_list = []
for i, (ff, y, x) in enumerate(zip(ff_list, y_list, x_list)):
    if pd.isnull(ff):
        rmse = float('inf')
    else:
        ff_coeff, num_coeffs = apply_coeffs(ff)
        f_hat = get_f(ff_coeff)
        coeffs, rmse = regression(f_hat, y, num_coeffs, np.array(x))
    rmse_list.append(rmse)
    print(rmse_list[-1])

pd.DataFrame(rmse_list).to_csv('02_rmse.csv', index=False, header=None)
