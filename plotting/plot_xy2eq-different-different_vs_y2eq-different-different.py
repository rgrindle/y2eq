"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 12, 2021

PURPOSE: Compare y2eq-fixed-different with y2eq-different-different.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def remove_nan_inf(data):
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]
    return data


# y2eq-different-different
rmse_y2eq_different_different = pd.read_csv('../eval_y2eq-different-different/02_rmse.csv')['rmse_ext'].values
rmse_y2eq_different_different = remove_nan_inf(rmse_y2eq_different_different)

# y2eq-fixed-different
rmse_xy2eq_different_different = pd.read_csv('../eval_xy2eq-different-different/02_rmse.csv')['rmse_ext'].values
rmse_xy2eq_different_different = remove_nan_inf(rmse_xy2eq_different_different)

# make figure
plt.figure()
plot_cdf(rmse_xy2eq_different_different, labels=False, color='C1', linestyle='dashed', linewidth=3, label='xy2eq-different-different')
plot_cdf(rmse_y2eq_different_different, labels=False, color='C0', linestyle='dashed', linewidth=3, label='y2eq-different-different')

plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.legend()
plt.xscale('log')
plt.ylim([0, 1000])
plt.savefig('012_fig1.pdf')

plt.xscale('linear')
plt.xlim([0, 3])
plt.savefig('012_fig1_zoom.pdf')
