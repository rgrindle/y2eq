"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate xy2eq-untrained-different and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_nn

import torch
import numpy as np

import json

torch.manual_seed(2)
np.random.seed(2)

with open('00_x_list.json', 'r') as json_file:
    x_list = np.array(json.load(json_file))

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = np.array(json.load(json_file))[:, :, 0]

input_list = np.stack((x_list, y_int_list), axis=2)

eval_nn(input_list=input_list,
        model_filename=None,
        INPUT_DIM=2)
