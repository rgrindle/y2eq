"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 16, 2021

PURPOSE: Compare y2eq-different-different-1000,
                 y2eq-fixed-different-1000, and
                 y2eq-fixed-different-30.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats


def remove_nan_inf(data):
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]
    return data


beam1 = pd.read_csv('02_rmse_beam1.csv')['rmse_ext'].values
beam1 = remove_nan_inf(beam1)

beam2 = pd.read_csv('02_rmse_beam2.csv')['rmse_ext'].values
beam2 = remove_nan_inf(beam2)

beam3 = pd.read_csv('02_rmse_beam3.csv')['rmse_ext'].values
beam3 = remove_nan_inf(beam3)

# make figure
fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for ax in axes:
    plt.sca(ax)
    plot_cdf(beam1, labels=False, label='1')
    plot_cdf(beam2, labels=False, label='2')
    plot_cdf(beam3, labels=False, label='3')

plt.sca(axes[1])
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.legend(loc='lower right', title='Beam size')
plt.xscale('log')
plt.ylim([0, 1000])

plt.sca(axes[0])
plt.legend(loc='lower right', title='Beam size')
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.xscale('linear')
plt.xlim([0, 3])

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig('ec025.pdf')
