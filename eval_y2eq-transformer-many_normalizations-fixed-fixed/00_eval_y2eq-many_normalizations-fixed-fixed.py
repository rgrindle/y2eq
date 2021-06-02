"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Get x and y to be used to evaluate y2eq-mb-fixed-fixed.

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.utils.eval import get_normal_x
from srvgd.utils.normalize import normalize

import numpy as np
import pandas as pd

import json


def write_x_y_lists_many_normalizations(eq_list_filename, x_type):
    assert x_type in ('fixed', 'different', 'normal')

    np.random.seed(0)

    eq_list = pd.read_csv(eq_list_filename, header=None).values.flatten()

    if x_type == 'fixed':
        x_int = np.arange(0.1, 3.1, 0.1)

    x_int_fixed = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    x_list = []
    y_int_normalized1_list = []
    y_int_unnormalized_list = []
    y_int_fixed_unnormalized_list = []
    y_ext_unnormalized_list = []
    for i, eq in enumerate(eq_list):

        if x_type == 'different':
            x_int = np.random.uniform(0.1, 3.1, 30)
            x_int.sort()
        elif x_type == 'normal':
            x_int = get_normal_x(num_points=30)
            x_int.sort()

        x_list.append(x_int.tolist())

        eq = EquationInfix(eq, apply_coeffs=False)
        y_int = eq.f(x_int)

        if np.any(np.isnan(y_int)):
            print('found nan')
            exit()

        y_int_unnormalized_list.append(y_int.tolist())
        y_int_fixed_unnormalized_list.append(eq.f(x_int_fixed).tolist())
        y_int_normalized1_list.append(normalize(y_int)[:, None].tolist())
        y_ext_unnormalized_list.append(eq.f(x_ext).tolist())

    # normalize2
    normalization_params = pd.read_csv('../datasets/normalization_params_many_normalizations.csv', header=None).values
    min_ = normalization_params[0, 0]
    scale_ = normalization_params[1, 0]
    print(normalization_params)
    y_int_normalized2_list = normalize(y_int_unnormalized_list,
                                       min_=min_,
                                       scale_=scale_)

    assert np.all(0 <= y_int_normalized2_list)
    assert np.all(y_int_normalized2_list <= 1)
    y_int_normalized2_list = y_int_normalized2_list[:, :, None].tolist()

    print(len(x_list))
    assert len(x_list) == len(y_int_normalized1_list)
    assert len(x_list) == len(y_int_normalized2_list)
    assert len(x_list) == len(y_int_unnormalized_list)
    assert len(x_list) == len(y_ext_unnormalized_list)

    with open('00_x_list.json', 'w') as file:
        json.dump(x_list, file)

    with open('00_y_int_normalized1_list.json', 'w') as file:
        json.dump(y_int_normalized1_list, file)

    with open('00_y_int_normalized2_list.json', 'w') as file:
        json.dump(y_int_normalized2_list, file)

    with open('00_y_int_unnormalized_list.json', 'w') as file:
        json.dump(y_int_unnormalized_list, file)

    with open('00_y_int_fixed_unnormalized_list.json', 'w') as file:
        json.dump(y_int_fixed_unnormalized_list, file)

    with open('00_y_ext_unnormalized_list.json', 'w') as file:
        json.dump(y_ext_unnormalized_list, file)


write_x_y_lists_many_normalizations('../datasets/equations_with_coeff_test_ff1000.csv',
                                    x_type='fixed')
