"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Test how normally picked x-values effect
         RMSE of y2eq.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


for int_ext_key in ['rmse_int', 'rmse_ext']:
    y2eq = {'fixed': pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse_105.csv')[int_ext_key],
            'normal (simga only)': pd.read_csv('../eval_y2eq-fixed-normal/02_rmse_sigma_only.csv')[int_ext_key],
            'normal (mu and sigma)': pd.read_csv('../eval_y2eq-fixed-normal/02_rmse_mu_and_sigma.csv')[int_ext_key]}

    for key in y2eq:
        y2eq[key] = y2eq[key][~np.isnan(y2eq[key])]
        y2eq[key] = y2eq[key][~np.isinf(y2eq[key])]

    fig, axes = plt.subplots(ncols=2, sharey=True,
                             figsize=(2*6.4, 4.8))

    for ax in axes:
        plt.sca(ax)
        for key in y2eq:
            plot_cdf(y2eq[key], labels=False, label='y2eq-fixed-'+key)

    if 'int' in int_ext_key:
        xlabel = 'RMSE on interpolation region ($x = [0.1, 0.2, \\cdots, 3.1)$)'
    else:
        xlabel = 'RMSE on extrapolation region ($x = [3.1, 3.2, \\cdots, 6.1)$)'

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

    plt.savefig('plot_y2eq_normal_tests_{}.pdf'.format(int_ext_key))

    results = mannwhitneyu(y2eq['fixed'],
                           y2eq['normal (mu and sigma)'],
                           alternative='less')
    print(key, results)
