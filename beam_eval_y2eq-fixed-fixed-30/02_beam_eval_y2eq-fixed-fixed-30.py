from srvgd.eval_scripts.get_rmse import get_rmse

import numpy as np
import pandas as pd

import json

with open('00_x_int_list.json', 'r') as json_file:
    x_int_list = json.load(json_file)

with open('00_y_int_unnormalized_list.json', 'r') as json_file:
    y_int_list = json.load(json_file)

with open('00_y_ext_unnormalized_list.json', 'r') as json_file:
    y_ext_list = json.load(json_file)

ff_list = pd.read_csv('01_predicted_ff_beam2.csv', header=None).values.flatten()

x_ext = np.arange(3.1, 6.1, 0.1)

get_rmse(x_int_list=x_int_list,
         y_int_list=y_int_list,
         x_ext=x_ext,
         y_ext_list=y_ext_list,
         ff_list=ff_list)
