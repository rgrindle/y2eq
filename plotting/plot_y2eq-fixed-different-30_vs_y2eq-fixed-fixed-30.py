"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 16, 2021

PURPOSE: Compare y2eq-fixed-different-30 with y2eq-fixed-fixed-30.

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


# y2eq-fixed-different-30
rmse_y2eq_fixed_different_30_ = pd.read_csv('../eval_y2eq-fixed-different/02_rmse_105.csv')['rmse_ext'].values
rmse_y2eq_fixed_different_30 = remove_nan_inf(rmse_y2eq_fixed_different_30_)

# y2eq-fixed-fixed-30
rmse_y2eq_fixed_fixed_30_ = pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse_105.csv')['rmse_ext'].values
rmse_y2eq_fixed_fixed_30 = remove_nan_inf(rmse_y2eq_fixed_fixed_30_)

# make figure
fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for ax in axes:
    plt.sca(ax)
    plot_cdf(rmse_y2eq_fixed_different_30, labels=False, color='#8B94FC', linestyle='dashed', linewidth=1, label='y2eq-fixed-different-30')
    plot_cdf(rmse_y2eq_fixed_fixed_30, labels=False, color='#8B94FC', linestyle='solid', linewidth=1, label='y2eq-fixed-fixed-30')

plt.sca(axes[1])
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.legend(loc='lower right')
plt.xscale('log')
plt.ylim([0, 1000])

plt.sca(axes[0])
plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
plt.ylabel('Cumulative counts')
plt.legend(loc='lower right')
plt.xscale('linear')
plt.xlim([0, 3])

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig('03_the_problem.pdf')


def replace_nan_with_inf(data):
    data = np.array(data)
    nan_indices = np.isnan(data)
    num_nan = np.sum(nan_indices)
    data[nan_indices] = [np.inf]*num_nan
    return data


rmse_y2eq_fixed_fixed_30_ = replace_nan_with_inf(rmse_y2eq_fixed_fixed_30_)
rmse_y2eq_fixed_different_30_ = replace_nan_with_inf(rmse_y2eq_fixed_different_30_)

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_fixed_30_,
                                   rmse_y2eq_fixed_different_30_,
                                   alternative='less')
print(results)
