"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 4, 2021

PURPOSE: Evaluate y2eq-transformer-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq_transformer

import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_y2eq_transformer(input_list=y_int_list,
                      model_filename='BEST_y2eq_transformer_pad.pt')
