"""
AUTHOR: Ryan Grindle

LAST MOFIFIED: Jan 18, 2021

PURPOSE: Get RMSE errors for CDF plot

NOTES: Requires 01_valid_eq....json exists. This
       contains a dictionary where the keys are indices
       and the values are equations.

TODO:
"""
from srvgd.utils.eval import fit_eq, normalize
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401
from eqlearner.dataset.processing.tokenization import get_string

import json
import torch
import numpy as np
import pandas as pd

# Get valid equations
file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
with open('01_valid_eq{}.json'.format(file_endname), 'r') as json_file:
    valid_equations = json.load(json_file)

for key in valid_equations:
    if 'x' not in valid_equations[key]:
        valid_equations[key] += '+0*x'

# Get expected y-values
test_data = torch.load('test_data_int_comp.pt', map_location=torch.device('cpu'))
y_true = np.array([d[0].tolist() for d in test_data])
eq_true = [get_string(d[1].tolist())[5:-3] for d in test_data]
print(y_true.shape)

# index = 0
# keys = list(valid_equations.keys())
# while len(valid_equations) > 3:
#     del valid_equations[keys[index]]
#     index += 1

x_int = np.arange(0.1, 3.1, 0.1)[:, None]
x_ext = np.arange(3.1, 6.1, 0.1)[:, None]

y_list = [y_true[int(i)] for i in valid_equations]
_, pred_rmse_list, pred_f_list = fit_eq(eq_list=valid_equations.values(),
                                        support=x_int,
                                        y_list=y_list)

eq_list = [eq_true[int(i)] for i in valid_equations]
_, true_rmse_list, true_f_list = fit_eq(eq_list=eq_list,
                                        support=x_int,
                                        y_list=y_list)

ext_rmse_list = []
for true_f, pred_f in zip(true_f_list, pred_f_list):
    _, pred_min_, pred_scale = normalize(pred_f(x_int), return_params=True)
    _, true_min_, true_scale = normalize(true_f(x_int), return_params=True)

    true_y = normalize(true_f(x_ext), true_min_, true_scale).flatten()
    pred_y = normalize(pred_f(x_ext), pred_min_, pred_scale).flatten()
    ext_rmse = np.sqrt(np.mean(np.power(true_y-pred_y, 2)))
    ext_rmse_list.append(ext_rmse)
print(ext_rmse_list)
pd.DataFrame([pred_rmse_list, ext_rmse_list]).T.to_csv('02_rmse{}.csv'.format(file_endname), index=False, header=['interpolated_rmse', 'extraplolated_rmse'])
