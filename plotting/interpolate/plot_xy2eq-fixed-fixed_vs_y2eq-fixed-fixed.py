"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 12, 2021

PURPOSE: Compare xy2eq-fixed-fixed with y2eq-fixed-fixed.

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


# xy2eq-fixed-fixed
rmse_xy2eq_fixed_fixed = pd.read_csv('../../eval_xy2eq-fixed-fixed/02_rmse.csv')['rmse_int'].values
rmse_xy2eq_fixed_fixed = remove_nan_inf(rmse_xy2eq_fixed_fixed)

# plot2eq-fixed-fixed
rmse_y2eq_fixed_fixed = pd.read_csv('../../eval_y2eq-fixed-fixed/02_rmse.csv')['rmse_int'].values
rmse_y2eq_fixed_fixed = remove_nan_inf(rmse_y2eq_fixed_fixed)

# make figure
plt.figure()
plot_cdf(rmse_xy2eq_fixed_fixed, labels=False, color='C1', linestyle='dashed', linewidth=3, label='xy2eq-fixed-fixed')
plot_cdf(rmse_y2eq_fixed_fixed, labels=False, color='C0', linestyle='solid', linewidth=3, label='y2eq-fixed-fixed')

plt.xlabel('RMSE on interpolation region ($x \\sim $uniform$\\left([0.1, 3.1)\\right)$)')
plt.ylabel('Cumulative counts')
plt.legend()
plt.xscale('log')
plt.ylim([0, 1000])
plt.savefig('020_fig1_interpolation.pdf')

plt.xscale('linear')
plt.xlim([0, 3])
plt.savefig('020_fig1_zoom_interpolation.pdf')
