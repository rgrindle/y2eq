"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Evaluate y2eq-mb-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.utils.eval import get_eq_y2eq_transformer
from srvgd.architecture.transformer.y2eq_transformer_mb import y2eq_trans_model

import torch
import numpy as np
import pandas as pd

import json

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_list = np.array(json.load(json_file))

with open('00_mb_normalized_list.json', 'r') as json_file:
    mb_normalized_list = np.array(json.load(json_file))[:, None]

mb_normalized_list = np.repeat(mb_normalized_list, 30, axis=1)
input_list = np.concatenate((y_int_list, mb_normalized_list), axis=-1)
input_list = input_list.tolist()

model_filename = 'BEST_y2eq_transformer_mb_30.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dict = torch.load('../models/'+model_filename, map_location=device)
y2eq_trans_model.load_state_dict(model_dict['state_dict'])

predicted_data = []
for i, input_ in enumerate(input_list):

    predicted = get_eq_y2eq_transformer(sentence=input_,
                                        model=y2eq_trans_model,
                                        device=device)
    predicted_data.append(predicted[5:-3])
    print(i, predicted_data[-1])

pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)
