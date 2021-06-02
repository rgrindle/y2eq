"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Try to fit a exp(x)^3 + exp(x)^2 + exp(x) + 1
         to all equations in dataset_test_ff1000.

NOTES:

TODO:
"""
from srvgd.utils.eval import fit_coeffs_and_get_rmse_vacc

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True,
                    help='Index to compute RMSE for')
args = parser.parse_args()

with open('00_y_int_fixed_unnormalized_list.json', 'r') as json_file:
    y_int_fixed_unnormalized_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_unnormalized_list = json.load(json_file)

ff_list = ['exp(x)**3 + exp(x)**2 + exp(x) + 1']*len(y_int_fixed_unnormalized_list)

fit_coeffs_and_get_rmse_vacc(args.index,
                             y_int_fixed_unnormalized_list,
                             y_ext_unnormalized_list,
                             ff_list)
