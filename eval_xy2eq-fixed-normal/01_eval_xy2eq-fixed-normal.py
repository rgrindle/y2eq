"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate xy2eq-fixed-normal and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq

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

eval_y2eq(input_list=input_list,
          model_filename='xy2eq_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900.pt',
          INPUT_DIM=2,
          OUTPUT_DIM=22)
