"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 2, 2021

PURPOSE: Evaluate y2eq-transformer-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq
from srvgd.utils.normalize import normalize
from srvgd.utils.rmse import RMSE

import numpy as np
import pandas as pd

import json


with open('01_predicted_y.json', 'r') as json_file:
    y_pred_list = np.squeeze(json.load(json_file))

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_normalized_list = json.load(json_file)


rmse_int_list = []
for i, (y_pred, y_int) in enumerate(zip(y_pred_list, y_int_normalized_list)):
    y_int = np.array(y_int).flatten()
    rmse_int = RMSE(y_int, y_pred)

    rmse_int_list.append(rmse_int)
    print(i, rmse_int_list[-1])

pd.DataFrame(rmse_int_list).to_csv('02_rmse.csv', index=False, header=['rmse_int'])
