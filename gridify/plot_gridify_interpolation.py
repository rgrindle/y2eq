from srvgd.plotting.cdf import plot_cdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rmse_data_numeric = pd.read_csv('../numeric_regression/rmse/_ff1000_with_x/all_data.csv')

rmse_data_gridify = pd.read_csv('../gridify/rmse/_ff1000_with_x/all_data.csv')

rmse_int = rmse_data_numeric['train_rmse'].values
rmse_ext = rmse_data_gridify['ext_normalized_rmse'].values
print(rmse_int.shape, rmse_ext.shape)
indices = [i for i, x in enumerate(rmse_int) if x < 0.001]
print(indices)

plt.figure()
plot_cdf(rmse_int, color='C0', label='all')
plot_cdf(rmse_int[indices], color='C1', label='best')

plt.xscale('log')
plt.xlabel('RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.legend()
plt.title('Figure 4a: RMSE of numeric NN\'s on interpolation region')

plt.figure()
rmse_ext_best = rmse_ext[indices]
rmse_ext_best = rmse_ext_best[~np.isnan(rmse_ext_best)]
plot_cdf(rmse_ext[~np.isnan(rmse_ext)], label='all')
plot_cdf(rmse_ext_best, label='best')
plt.xscale('log')
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.title('Figure 4b: RMSE of symbolic NN using gridify\ntechnique on extrapolation region')
plt.legend()

plt.show()
