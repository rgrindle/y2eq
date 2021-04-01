"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 22, 2021

PURPOSE: Compare y2eq-fixed-normal-30 with y2eq-fixed-fixed-30.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats


def remove_nan_inf(data):
    for key in ['rmse_int', 'rmse_ext']:
        data[key] = data[key][~np.isnan(data[key])]

    return data


# y2eq-fixed-normal-30
rmse_y2eq_fixed_normal_30_ = pd.read_csv('../eval_y2eq-fixed-normal/02_rmse.csv')
rmse_y2eq_fixed_normal_30 = remove_nan_inf(rmse_y2eq_fixed_normal_30_)

# y2eq-fixed-fixed-30
rmse_y2eq_fixed_fixed_30_ = pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse_105.csv')
rmse_y2eq_fixed_fixed_30 = remove_nan_inf(rmse_y2eq_fixed_fixed_30_)


def make_figure(extrap):
    if extrap:
        xlabel = 'RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)'
        key = 'rmse_ext'
    else:
        xlabel = 'RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)'
        key = 'rmse_int'

    fig, axes = plt.subplots(ncols=2, sharey=True,
                             figsize=(2*6.4, 4.8))

    for ax in axes:
        plt.sca(ax)
        plot_cdf(rmse_y2eq_fixed_normal_30[key][~np.isnan(rmse_y2eq_fixed_normal_30[key])], labels=False, color='#8B94FC', linestyle='dashed', linewidth=1, label='y2eq-fixed-normal-30')
        plot_cdf(rmse_y2eq_fixed_fixed_30[key][~np.isnan(rmse_y2eq_fixed_fixed_30[key])], labels=False, color='#8B94FC', linestyle='solid', linewidth=1, label='y2eq-fixed-fixed-30')

    plt.sca(axes[1])
    plt.xlabel(xlabel)
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim([0, 1000])

    plt.sca(axes[0])
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative counts')
    plt.legend(loc='lower right')
    plt.xscale('linear')
    plt.xlim([0, 3])

    plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)


# make figure
make_figure(extrap=False)
plt.savefig('y2eq_normal_test_interp.pdf')

make_figure(extrap=True)
plt.savefig('y2eq_normal_test_extrap.pdf')


def replace_nan_with_inf(data):
    data = np.array(data)
    nan_indices = np.isnan(data)
    num_nan = np.sum(nan_indices)
    data[nan_indices] = [np.inf]*num_nan
    return data


rmse_y2eq_fixed_fixed_30_ = replace_nan_with_inf(rmse_y2eq_fixed_fixed_30_['rmse_ext'])
rmse_y2eq_fixed_normal_30_ = replace_nan_with_inf(rmse_y2eq_fixed_normal_30_['rmse_ext'])

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_fixed_30_,
                                   rmse_y2eq_fixed_normal_30_,
                                   alternative='less')
print(results)
