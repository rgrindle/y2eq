"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate y2eq-untrained-normal and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_nn

import torch
import numpy as np

import json

torch.manual_seed(1234)
np.random.seed(1234)

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_nn(input_list=y_int_list,
        model_filename=None)
