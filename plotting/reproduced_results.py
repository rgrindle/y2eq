"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 16, 2021

PURPOSE: Show that I have successfully recreated model
         from "A Seq2Seq approach to Symbolic Regression"

NOTES:

TODO:
"""

from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


def remove_nan_inf(data):
    for key in data:
        data[key] = data[key][~np.isnan(data[key])]
        data[key] = data[key][~np.isinf(data[key])]
    return data


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # y2eq-fixed-fixed-30
    y2eq_data = pd.read_csv('../eval_y2eq-fixed-fixed/02_rmse.csv')
    rmse_y2eq_fixed_fixed_30 = {'int': y2eq_data['rmse_int'].values,
                                'ext': y2eq_data['rmse_ext'].values}

    # numeric regression NN
    numeric_data = pd.read_csv('../numeric_regression/rmse/_ff1000/all_data.csv')
    rmse_numeric_regression_nn = {'int': numeric_data['train_rmse'].values,
                                  'ext': numeric_data['test_rmse'].values}

    # genetic programming
    gp_data = pd.read_csv('../src/gp/gp_data/ff1000_100/all_rmse_data.csv')
    rmse_gp = {'int': gp_data['train_normalized_rmse'].values,
               'ext': gp_data['test_normalized_rmse'].values}

    rmse_y2eq_fixed_fixed_30 = remove_nan_inf(rmse_y2eq_fixed_fixed_30)
    rmse_numeric_regression_nn = remove_nan_inf(rmse_numeric_regression_nn)
    rmse_gp = remove_nan_inf(rmse_gp)

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
    plt.figure()
    plot_cdf(rmse_y2eq_fixed_fixed_30['ext'], labels=False, color='#8B94FC', label='y2eq-fixed-fixed-30')
    plot_cdf(rmse_numeric_regression_nn['ext'], labels=False, color='#CF6875', label='Numeric NN')
    plot_cdf(rmse_gp['ext'], labels=False, color='C8', label='Genetic programming')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1000])
    plt.savefig('reproduce_results_extrapolation.pdf')

    plt.xscale('linear')
    plt.xlim([0, 3])
    plt.savefig('reproduce_results_extrapolation_zoom.pdf')
