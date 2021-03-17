"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 16, 2021

PURPOSE: Plot for EC026.0

NOTES: Compare y2eq-fixed-fixed-30 and xy2eq-fixed-fixed-30
       when using the weights from y2eq-fixed-fixed-30.

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    def remove_nan_inf(data):
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        return data

    rmse_xy2eq_fixed_fixed = pd.read_csv('02_rmse_weights.csv')['rmse_ext'].values.flatten()
    rmse_xy2eq_fixed_fixed = remove_nan_inf(rmse_xy2eq_fixed_fixed)

    rmse_y2eq_fixed_fixed = pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse_105.csv')['rmse_ext'].values.flatten()
    rmse_y2eq_fixed_fixed = remove_nan_inf(rmse_y2eq_fixed_fixed)

    # make figure
    fig, axes = plt.subplots(ncols=2, sharey=True,
                             figsize=(2*6.4, 4.8))

    for ax in axes:
        plt.sca(ax)
        plot_cdf(rmse_xy2eq_fixed_fixed, labels=False, linewidth=5, label='xy2eq-fixed-fixed (weights from y2eq)')
        plot_cdf(rmse_y2eq_fixed_fixed, labels=False, linewidth=1, label='y2eq-fixed-fixed')

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
    plt.savefig('03_xy2eq_weights.pdf')
