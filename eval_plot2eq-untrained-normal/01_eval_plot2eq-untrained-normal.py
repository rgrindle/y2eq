"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate plot2eq-untrained-normal and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_plot2eq
from srvgd.utils.get_data_image import get_data_image

import torch
import numpy as np

import json

torch.manual_seed(1234)
np.random.seed(1234)

with open('00_x_list.json', 'r') as json_file:
    x_list = np.array(json.load(json_file))

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = np.array(json.load(json_file))[:, :, 0]

images = []
for x, y in zip(x_list, y_int_list):
    img = get_data_image(np.vstack((x, y)).T, bins=(64, 64))
    images.append(img)

eval_plot2eq(input_list=images,
             model_filename=None)
