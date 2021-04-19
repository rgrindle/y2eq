"""
AUTHOR: Ryan Grindle

LAST MODIFIED: April 2, 2021

PURPOSE: Get RMSE after fitting coefficients.

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import get_f, apply_coeffs, RMSE, is_eq_valid
import pandas as pd
import numpy as np
from icecream import ic

import json

from scipy.optimize import minimize


def replace_nan(array, replace_value=1.):
    array[np.isnan(array)] = replace_value
    return array


import cma


def regression_cmaes(f_hat, y, num_coeffs, support):
    def loss(c, x):
        y_hat = f_hat(c, x).flatten()
        assert np.array(y_hat).shape == np.array(y).shape
        return RMSE(y_hat, y)

    es = cma.CMAEvolutionStrategy(num_coeffs * [0], 0.5,
                                  {'bounds': [-3, 3],
                                   # consistent if np.random.seed is set
                                   'seed': np.random.randint(1, 10000000)})
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [loss(c, support) for c in solutions])
    coeffs = es.best.x
    rmse = es.best.f
    return coeffs, rmse


def regression(f_hat, y, num_coeffs, support):
    def loss(c, x):
        y_hat = f_hat(c, x).flatten()
        # y_hat = replace_nan(y_hat)
        # if np.any(np.isnan(y)):
        #     print('y has nan! WHAT!>!>!>')
        #     exit()
        # if np.any(np.isnan(y_hat)):
        #     ic(c)
        #     ic(f_hat(c, x))
        #     print('y_hat has nan! WHAT!>!>!>')
        #     exit()
        return RMSE(y_hat, y)

    rmse_list = []
    coeffs_list = []
    for _ in range(10):
        res = minimize(loss, np.random.uniform(-3, 3, num_coeffs), args=(support,),
                       bounds=[(-3, 3)]*num_coeffs,
                       method='L-BFGS-B')
        rmse_list.append(loss(res.x, support))
        coeffs_list.append(res.x)
    # assert len(np.unique(rmse_list)) == 1
    index = np.nanargmin(rmse_list)
    return coeffs_list[index], rmse_list[index]


with open('00_x_list.json', 'r') as json_file:
    x_list = json.load(json_file)

with open('00_y_unnormalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_list = json.load(json_file)

ff_list = pd.read_csv('00_ff_list.csv', header=None).values.flatten()

x_ext = np.arange(3.1, 6.1, 0.1)

rmse_int_list = []
rmse_ext_list = []
for i, (ff, y, x, y_ext) in enumerate(zip(ff_list, y_list, x_list, y_ext_list)):
    if pd.isnull(ff):
        rmse_int = float('inf')
        rmse_ext = float('inf')
        print('null')
    elif is_eq_valid(ff):
        ic(ff)
        ff_coeff, num_coeffs = apply_coeffs(ff)
        f_hat = get_f(ff_coeff)
        coeffs, rmse_int = regression_cmaes(f_hat, y, num_coeffs, np.array(x))

        y_int_true_norm, true_min_, true_scale = normalize(y, return_params=True)
        y_int_pred_norm = normalize(f_hat(c=coeffs, x=np.array(x)), true_min_, true_scale)

        y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
        y_ext_pred_norm = normalize(f_hat(c=coeffs, x=x_ext), true_min_, true_scale)

        print('')
        rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)
        rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)

        if np.isnan(rmse_int):
            if 'log' in ff:
                print('nan possibly caused by log')
            else:
                print('??????????????????????????????????')
    else:
        rmse_int = float('inf')
        rmse_ext = float('inf')

    rmse_ext_list.append(rmse_ext)
    rmse_int_list.append(rmse_int)
    print(i, rmse_int_list[-1], rmse_ext_list[-1])

pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('01_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])
