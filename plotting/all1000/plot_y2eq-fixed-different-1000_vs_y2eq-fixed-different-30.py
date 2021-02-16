"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 12, 2021

PURPOSE: Compare y2eq-fixed-different-1000 with y2eq-different-different-30.

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


# y2eq-fixed-different-1000
rmse_y2eq_fixed_different_1000 = pd.read_csv('../../eval_y2eq-fixed-different-1000/02_rmse.csv', header=None).values
rmse_y2eq_fixed_different_1000 = remove_nan_inf(rmse_y2eq_fixed_different_1000)

# y2eq-fixed-different-30
rmse_y2eq_fixed_different_30 = pd.read_csv('../../eval_y2eq-fixed-different/02_rmse.csv')['rmse_ext'].values
rmse_y2eq_fixed_different_30 = remove_nan_inf(rmse_y2eq_fixed_different_30)

# make figure
plt.figure()
plot_cdf(rmse_y2eq_fixed_different_1000, labels=False, color='C0', linestyle='solid', linewidth=3, label='y2eq-fixed-different-1000')
plot_cdf(rmse_y2eq_fixed_different_30, labels=False, color='C0', linestyle='dashed', linewidth=3, label='y2eq-fixed-different-30')

plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.legend()
plt.xscale('log')
plt.ylim([0, 1000])
plt.savefig('013_fig1.pdf')

plt.xscale('linear')
plt.xlim([0, 3])
plt.savefig('013_fig1_zoom.pdf')
