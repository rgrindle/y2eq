"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 2, 2021

PURPOSE: Save x and y data for evaluation in next script.

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import get_f

import numpy as np

import json

np.random.seed(0)


def get_x_y(x_int_type, x_int_num, x_ext, eq_list,
            a=0.1, b=3.1):
    assert x_int_type in ('random', 'uniform')
    assert x_int_num > 0

    if x_int_type == 'uniform':
        x_int = np.arange(a, b, (b-a)/x_int_num)
        assert len(x_int) == x_int_num

    x_int_list = []
    y_int_normalized_list = []
    y_int_unnormalized_list = []
    y_ext_unnormalized_list = []
    for eq in eq_list:
        f = get_f(eq)
        if x_int_type == 'random':
            x_int = np.random.uniform(a, b, x_int_num)
            x_int.sort()
        x_int_list.append(x_int.tolist())
        y_int = f(x_int)
        y_int_unnormalized_list.append(y_int.tolist())
        y_int_normalized_list.append(normalize(y_int)[:, None].tolist())
        y_ext_unnormalized_list.append(f(x_ext).tolist())

    with open('00_x_int_list.json', 'w') as file:
        json.dump(x_int_list, file)

    with open('00_y_int_normalized_list.json', 'w') as file:
        json.dump(y_int_normalized_list, file)

    with open('00_y_int_unnormalized_list.json', 'w') as file:
        json.dump(y_int_unnormalized_list, file)

    with open('00_y_ext_unnormalized_list.json', 'w') as file:
        json.dump(y_ext_unnormalized_list, file)
