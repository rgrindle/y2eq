
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # fixed x
    file_endname = '_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900'
    rmse_data_fixed = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]

    # random x
    rmse_data_random = pd.read_csv('02_rmse.csv').values.flatten()

    # make figure
    plt.figure()
    rmse_data_fixed = rmse_data_fixed[~np.isnan(rmse_data_fixed)]
    rmse_data_fixed = rmse_data_fixed[~np.isinf(rmse_data_fixed)]

    rmse_data_random = rmse_data_random[~np.isnan(rmse_data_random)]
    rmse_data_random = rmse_data_random[~np.isinf(rmse_data_random)]

    plot_cdf(rmse_data_fixed, labels=False, ymax=1., color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random, labels=False, ymax=1., color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig('03_eval_y2eq-fixed-different_ymax1.pdf')

    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.ylim([0, 1000])
    plt.legend()
    plt.xscale('log')
    plt.savefig('03_eval_y2eq-fixed-different_ymaxauto.pdf')

    # make figure x in [0, 3]
    plt.xscale('linear')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('03_eval_y2eq-fixed-different.pdf')
