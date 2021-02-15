"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 12, 2021

PURPOSE: Compare xy2eq-different-different with plot2eq-different-different.

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


# xy2eq-different-different
rmse_xy2eq_different_different = pd.read_csv('../eval_xy2eq-different-different/02_rmse.csv', header=None).values
rmse_xy2eq_different_different = remove_nan_inf(rmse_xy2eq_different_different)

# plot2eq-different-different
rmse_plot2eq_different_different = pd.read_csv('../eval_plot2eq-different-different/02_rmse.csv', header=None).values
rmse_plot2eq_different_different = remove_nan_inf(rmse_plot2eq_different_different)

# make figure
plt.figure()
plot_cdf(rmse_xy2eq_different_different, labels=False, label='xy2eq-different-different')
plot_cdf(rmse_plot2eq_different_different, labels=False, label='plot2eq-different-different')

plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.legend()
plt.xscale('log')
plt.ylim([0, 1000])
plt.savefig('010_fig1.pdf')

plt.xscale('linear')
plt.xlim([0, 3])
plt.savefig('010_fig1_zoom.pdf')
