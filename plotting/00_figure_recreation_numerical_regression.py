"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 14, 2021

PURPOSE: Gather rmse data from multiple files in
         SRvGD/src/srvgd/numerical_regression/rmse/
         and combine data into a single file. Next,
         script will make a plot.

NOTES:

TODO:
"""

import pandas as pd

import os

path = os.path.join('..', 'src', 'srvgd', 'numerical_regression', 'rmse')

all_data = {'train_rmse': [],
            'train_unscaled_rmse': [],
            'test_rmse': [],
            'test_unscaled_rmse': []}

for i in range(1000):
    data = pd.read_csv(os.path.join(path, '{}.txt'.format(i)))
    for key in all_data:
        all_data[key].append(data[key].values[0])

save_file = os.path.join(path, 'all_data.csv')
pd.DataFrame(all_data).to_csv(save_file, index=False)
