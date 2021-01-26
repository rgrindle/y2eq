import numpy as np
import pandas as pd

file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
# file_endname = '_epochs100_0'
rmse_data = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]
print(len(rmse_data))
# mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))

eq_true = pd.read_csv('equations_with_coeff_test.csv'.format(file_endname), header=None).values.flatten()
print(eq_true.shape)

indices = np.argsort(rmse_data)

for i in indices:
    print(rmse_data[i], eq_true[i])
