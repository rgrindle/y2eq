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


# y2eq-different-different-1000
rmse_y2eq_fixed_fixed_1000_ = pd.read_csv('../eval_y2eq-fixed-fixed-1000/02_rmse_105.csv')['rmse_ext'].values
rmse_y2eq_fixed_fixed_1000 = remove_nan_inf(rmse_y2eq_fixed_fixed_1000_)


# y2eq-fixed-different-1000
rmse_y2eq_fixed_different_1000_ = pd.read_csv('../eval_y2eq-fixed-different-1000/02_rmse_105.csv')['rmse_ext'].values
rmse_y2eq_fixed_different_1000 = remove_nan_inf(rmse_y2eq_fixed_different_1000_)

# y2eq-fixed-different-30
rmse_y2eq_fixed_different_30_ = pd.read_csv('../eval_y2eq-fixed-different/02_rmse_105.csv')['rmse_ext'].values
rmse_y2eq_fixed_different_30 = remove_nan_inf(rmse_y2eq_fixed_different_30_)


# make figure
fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for ax in axes:
    plt.sca(ax)
    plot_cdf(rmse_y2eq_fixed_fixed_1000, labels=False, color='#8B94FC', linestyle='solid', linewidth=3, label='y2eq-fixed-fixed-1000')
    plot_cdf(rmse_y2eq_fixed_different_1000, labels=False, color='#8B94FC', linestyle='dashed', linewidth=3, label='y2eq-fixed-different-1000')
    plot_cdf(rmse_y2eq_fixed_different_30, labels=False, color='#8B94FC', linestyle='dashed', linewidth=1, label='y2eq-fixed-different-30')

plt.sca(axes[1])
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.legend(loc='lower right')
plt.xscale('log')
plt.ylim([0, 1000])

plt.sca(axes[0])
plt.legend(loc='lower right')
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.xscale('linear')
plt.xlim([0, 3])

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig('04_main_result.pdf')


def replace_nan_with_inf(data):
    data = np.array(data)
    nan_indices = np.isnan(data)
    num_nan = np.sum(nan_indices)
    data[nan_indices] = [np.inf]*num_nan
    return data


rmse_y2eq_fixed_different_30_ = replace_nan_with_inf(rmse_y2eq_fixed_different_30_)
rmse_y2eq_fixed_different_1000_ = replace_nan_with_inf(rmse_y2eq_fixed_different_1000_)
rmse_y2eq_fixed_fixed_1000_ = replace_nan_with_inf(rmse_y2eq_fixed_fixed_1000_)

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_different_1000_,
                                   rmse_y2eq_fixed_different_30_,
                                   alternative='less')
print('y2eq-fixed-different-1000 < y2eq-fixed-different-30', results)

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_different_1000_,
                                   rmse_y2eq_fixed_fixed_1000_,
                                   alternative='two-sided')
print('y2eq-fixed-different-1000 != y2eq-fixed-fixed-1000', results)
