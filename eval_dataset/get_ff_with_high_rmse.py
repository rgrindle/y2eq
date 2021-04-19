import pandas as pd
import numpy as np

rmse_list = pd.read_csv('01_rmse_all.csv')['rmse_int'].values
ff_list = pd.read_csv('00_ff_list.csv', header=None).values.flatten()
assert ff_list.shape == rmse_list.shape

indices = np.argsort(rmse_list)

for count, i in enumerate(reversed(indices)):
    if count > 10:
        break

    print(rmse_list[i])
    print(ff_list[i])
    print('')

    