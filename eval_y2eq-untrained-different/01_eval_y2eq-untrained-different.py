"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate y2eq-untrained-different and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq

import torch
import numpy as np

import json

torch.manual_seed(2)
np.random.seed(2)

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_y2eq(input_list=y_int_list,
          model_filename=None)
