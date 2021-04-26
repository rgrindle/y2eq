"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 13, 2021

PURPOSE: Compute root mean squared error of functional forms
         output by neural network

NOTES:

TODO:
"""
from srvgd.utils.eval import fit_coeffs_and_get_rmse

import pandas as pd

import json

with open('00_y_int_fixed_unnormalized_list.json', 'r') as json_file:
    y_int_fixed_unnormalized_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_unnormalized_list = json.load(json_file)

ff_list = pd.read_csv('01_predicted_ff.csv', header=None).values.flatten()

fit_coeffs_and_get_rmse(y_int_fixed_unnormalized_list,
                        y_ext_unnormalized_list,
                        ff_list)
