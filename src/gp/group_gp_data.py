"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: Gather rmse data from multiple files in
         SRvGD/src/gp/gp_data/
         and combine data into a single file.

NOTES:

TODO:
"""

import pandas as pd

import os

all_data = {'train_normalized_rmse': [],
            'train_rmse': [],
            'test_normalized_rmse': [],
            'test_rmse': []}

for i in range(1000):
    data = pd.read_csv(os.path.join('gp_data', 'rmse{}.txt'.format(i)))

    for key in all_data:
        all_data[key].append(data[key].values[0])

save_file = os.path.join('gp_data', 'all_rmse_data.csv')
pd.DataFrame(all_data).to_csv(save_file, index=False)
