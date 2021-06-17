"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 2, 2021

PURPOSE: Evaluate y2eq-transformer-fixed-fixed and save functional form.

NOTES:

TODO:
"""
import numpy as np
import matplotlib.pyplot as plt

import json


with open('01_predicted_y.json', 'r') as json_file:
    y_pred_list = np.squeeze(json.load(json_file))

with open('00_y_int_normalized_list.json', 'r') as json_file:
    y_int_normalized_list = np.squeeze(json.load(json_file))

x = np.arange(0.1, 3.1, 0.1)
for i, (y_pred, y_int) in enumerate(zip(y_pred_list, y_int_normalized_list)):

    plt.close('all')
    plt.figure()
    plt.plot(x, y_pred, '.-', label='pred')
    plt.plot(x, y_int, '.-', label='true')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.savefig('plots/{}.png'.format(i))
    print(i, 'done')
