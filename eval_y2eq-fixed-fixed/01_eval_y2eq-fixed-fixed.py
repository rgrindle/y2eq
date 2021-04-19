"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Evaluate y2eq-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_y2eq

import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_y2eq(input_list=y_int_list,
          OUTPUT_DIM=22,
          model_filename='cnn_dataset_train_ff1000_batchsize32_lr0.0001_clip1_layers10_105.pt')
