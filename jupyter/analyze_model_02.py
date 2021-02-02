"""
AUTHOR: Ryan Grindle

LAST MOFIFIED: Jan 28, 2021

PURPOSE: Get RMSE errors for CDF plot

NOTES: Requires 01_valid_eq....json exists. This
       contains a dictionary where the keys are indices
       and the values are equations.

TODO:
"""
from srvgd.utils.eval import fit_eq, normalize, get_f

import json
import torch
import numpy as np
import pandas as pd

import os

# Get valid equations
file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
# file_endname = '_epochs100_0'
with open('01_valid_eq{}.json'.format(file_endname), 'r') as json_file:
    valid_equations = json.load(json_file)

for key in valid_equations:
    if 'x' not in valid_equations[key]:
        valid_equations[key] += '+0*x'

# Get expected y-values
device = torch.device('cpu')
# test_data = torch.load('dataset_test_ff.pt', map_location=device)
# test_data = torch.load('test_data_int_comp.pt', map_location=device)
eq_true = pd.read_csv(os.path.join('..', 'datasets', 'equations_with_coeff_test_ff.csv'), header=None).values.flatten()

# y_true = np.array([d[0].tolist() for d in test_data])
# ff_true = [get_string(d[1].tolist())[5:-3] for d in test_data]
# print(y_true.shape)

# index = 0
# keys = list(valid_equations.keys())
# while len(valid_equations) > 3:
#     del valid_equations[keys[index]]
#     index += 1

x_int = np.arange(0.1, 3.1, 0.1)[:, None]
x_ext = np.arange(3.1, 6.1, 0.1)[:, None]

y_true = []
true_f_list = []
count = 0
for i in valid_equations:
    eq = eq_true[int(i)]
    f = get_f(eq)
    true_f_list.append(f)
    y = f(x_int).flatten()
    y_true.append(y)
    if np.any(np.isnan(f(x_ext))):
        # print(eq)
        # print(f(x_ext))
        count += 1
print('Number of true equations with nan on extrapolation region', count)

# y_list = []
# for i in valid_equations:
#     f = get_f(eq_true[int(i)])
#     y_list.append(f(x_int))
# print(np.around(normalize(f(x_int).flatten()), 7))
# print(np.around(y_true[int(i)], 7))
# assert np.all(np.around(normalize(f(x_int).flatten()), 7) == np.around(y_true[int(i)], 7))
# eq_c, num_coeffs = apply_coeffs(ff_true[int(i)])
# print(eq_c)
# f_hat = get_f(eq_c)
# # c = (1.072, 0.960, -1.133)
# # print(f_hat(c, x_int))
# coeffs, rmse = regression(f_hat, f(x_int).flatten(), num_coeffs, x_int)
# print(coeffs)
# print(rmse)
#     if int(i) == 3:
#         break
# exit()


# y_list = [y_true[int(i)] for i in valid_equations]
_, pred_rmse_list, pred_f_list = fit_eq(eq_list=valid_equations.values(),
                                        support=x_int,
                                        y_list=y_true)

# eq_list = [eq_true[int(i)] for i in valid_equations]
# _, true_rmse_list, true_f_list = fit_eq(eq_list=eq_list,
#                                         support=x_int,
#                                         y_list=y_list)

print(len(true_f_list), len(pred_f_list))
ext_rmse_list = []
for true_f, pred_f in zip(true_f_list, pred_f_list):
    # _, pred_min_, pred_scale = normalize(pred_f(x_int), return_params=True)
    _, true_min_, true_scale = normalize(true_f(x_int), return_params=True)

    true_y = normalize(true_f(x_ext), true_min_, true_scale).flatten()
    pred_y = normalize(pred_f(x_ext), true_min_, true_scale).flatten()
    ext_rmse = np.sqrt(np.mean(np.power(true_y-pred_y, 2)))
    ext_rmse_list.append(ext_rmse)
print(ext_rmse_list)
pd.DataFrame([list(valid_equations.keys()), pred_rmse_list, ext_rmse_list]).T.to_csv('02_rmse{}.csv'.format(file_endname), index=False, header=['index', 'interpolated_rmse', 'extraplolated_rmse'])
