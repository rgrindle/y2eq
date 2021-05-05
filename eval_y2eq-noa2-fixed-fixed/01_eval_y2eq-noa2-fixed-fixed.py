"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 4, 2021

PURPOSE: Evaluate y2eq-noa2-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq

import torch

import os
import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

train_data = torch.load(os.path.join('..', 'datasets', 'dataset_train_ff1000_with_coeffs_v2.pt'),
                        map_location=torch.device('cpu'))

eval_y2eq(input_list=y_int_list,
          model_filename='cnn_dataset_train_ff1000_with_coeffs_v2_batchsize32_lr0.0001_clip1_layers10_includecoeffsTrue_includecoeffvaluesTrue_105.pt',
          include_coeffs=True,
          include_coeff_values=True,
          DEC_MAX_LENGTH=len(train_data[0][1]))
