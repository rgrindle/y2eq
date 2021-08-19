"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 1, 2021

PURPOSE: Show that I have successfully recreated model
         from "A Seq2Seq approach to Symbolic Regression"

NOTES:

TODO:
"""

from srvgd.plotting.cdf import plot_cdf

import matplotlib.pyplot as plt
import scipy.stats


def remove_nan_inf(data):
    for key in data:
        data[key] = data[key][~np.isnan(data[key])]
        data[key] = data[key][~np.isinf(data[key])]
    return data


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # y2eq-fixed-fixed-30
    y2eq_data = pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse_150.csv')
    rmse_y2eq_fixed_fixed_30_ = {'int': y2eq_data['rmse_int'].values,
                                 'ext': y2eq_data['rmse_ext'].values}

    # numeric regression NN
    numeric_data = pd.read_csv('../numeric_regression/rmse/_ff1000/all_data.csv')
    rmse_numeric_regression_nn_ = {'int': numeric_data['train_rmse'].values,
                                   'ext': numeric_data['test_rmse'].values}

    # genetic programming
    gp_data = pd.read_csv('../src/gp/gp_data/ff1000_100/all_rmse_data.csv')
    rmse_gp_ = {'int': gp_data['train_normalized_rmse'].values,
                'ext': gp_data['test_normalized_rmse'].values}

    rmse_y2eq_fixed_fixed_30 = remove_nan_inf(rmse_y2eq_fixed_fixed_30_)
    rmse_numeric_regression_nn = remove_nan_inf(rmse_numeric_regression_nn_)
    rmse_gp = remove_nan_inf(rmse_gp_)

    for key in ['int', 'ext']:
        print(rmse_y2eq_fixed_fixed_30[key].shape)
        print(rmse_numeric_regression_nn[key].shape)
        print(rmse_gp[key].shape)

    # make interpolation figure
    plt.figure()
    plot_cdf(rmse_y2eq_fixed_fixed_30['int'], labels=False, color='#8B94FC', label='y2eq-fixed-fixed-30')
    plot_cdf(rmse_numeric_regression_nn['int'], labels=False, color='#CF6875', label='Numeric NN')
    plot_cdf(rmse_gp['int'], labels=False, color='C8', label='Genetic programming')

    plt.xlabel('RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1000])
    plt.savefig('reproduce_results_interpolation.pdf')

    plt.xscale('linear')
    plt.xlim([0, 3])
    plt.savefig('reproduce_results_interpolation_zoom.pdf')

    # make extrapolation figure
    fig, axes = plt.subplots(ncols=2, sharey=True,
                             figsize=(2*6.4, 4.8))

    for ax in axes:
        plt.sca(ax)
        plot_cdf(rmse_y2eq_fixed_fixed_30['ext'], labels=False, color='#8B94FC', linewidth=1, label='y2eq')
        plot_cdf(rmse_numeric_regression_nn['ext'], labels=False, color='#CF6875', label='Numeric NN')
        plot_cdf(rmse_gp['ext'], labels=False, color='C8', label='Genetic programming')

    plt.sca(axes[1])
    plt.xlabel('Numeric cost on extrapolation region ($x = \\left\\{3.1, 3.2, \\cdots, 6.0\\right\\}$)')
    # plt.ylabel('Cumulative counts')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim([0, 1000])
    # plt.savefig('reproduce_results_extrapolation.pdf')

    plt.sca(axes[0])
    plt.legend(loc='lower right')
    plt.xlabel('Numeric cost on extrapolation region ($x = \\left\\{3.1, 3.2, \\cdots, 6.0\\right\\}$)')
    plt.ylabel('Cumulative counts')
    plt.xscale('linear')
    plt.xlim([0, 3])
    # plt.savefig('reproduce_results_extrapolation_zoom.pdf')

    plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
    plt.savefig('reproduce_results_extrapolation.pdf')


def replace_nan_with_inf(data):
    for key in data:
        data[key] = np.array(data[key])
        nan_indices = np.isnan(data[key])
        num_nan = np.sum(nan_indices)
        data[key][nan_indices] = [np.inf]*num_nan
    return data


rmse_y2eq_fixed_fixed_30_ = replace_nan_with_inf(rmse_y2eq_fixed_fixed_30_)
rmse_numeric_regression_nn_ = replace_nan_with_inf(rmse_numeric_regression_nn_)
rmse_gp_ = replace_nan_with_inf(rmse_gp_)

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_fixed_30_['ext'],
                                   rmse_numeric_regression_nn_['ext'],
                                   alternative='less')
print('y2eq-fixed-fixed-30 < numeric regression', results)

results = scipy.stats.mannwhitneyu(rmse_y2eq_fixed_fixed_30_['ext'],
                                   rmse_gp_['ext'],
                                   alternative='less')
print('y2eq-fixed-fixed-30 < genetic programming', results)
