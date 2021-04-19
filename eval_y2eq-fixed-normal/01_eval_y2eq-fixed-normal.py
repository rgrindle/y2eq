"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 13, 2021

PURPOSE: Evaluate y2eq-fixed-normal and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import eval_nn

import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

eval_nn(input_list=y_int_list,
        model_filename='cnn_dataset_train_ff1000_batchsize32_lr0.0001_clip1_layers10_105.pt')
