"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 4, 2021

PURPOSE: Gather rmse data from multiple files in
         SRvGD/gridify/rmse/
         and combine data into a single file. Next,
         script will make a plot.

NOTES:

TODO:
"""

import pandas as pd

import os

dataset = '_ff1000_with_x'
all_data = {'int_normalized_rmse': [],
            'int_rmse': [],
            'ext_normalized_rmse': [],
            'ext_rmse': []}

missing = []
for i in range(1000):
    try:
        data = pd.read_csv(os.path.join('rmse', dataset, '{}.txt'.format(i)))
        for key in all_data:
            all_data[key].append(data[key].values[0])
    except FileNotFoundError:
        missing.append(i)

print(missing)

save_file = os.path.join('rmse', dataset, 'all_data.csv')
pd.DataFrame(all_data).to_csv(save_file, index=False)
