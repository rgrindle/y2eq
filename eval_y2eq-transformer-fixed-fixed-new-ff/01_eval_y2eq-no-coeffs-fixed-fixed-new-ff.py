"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Evaluate y2eq-no-coeffs-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq_transformer

import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_y2eq_transformer(input_list=y_int_list,
                      model_filename='BEST_y2eq_transformer.pt')
