
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # fixed x
    file_endname = '_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900'
    rmse_data_fixed = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]

    # random x - NN only y
    rmse_data_random_y = pd.read_csv('../fixed_x_problem/02_rmse.csv').values.flatten()

    # # random x - NN with x, y
    # rmse_data_random_xy200 = pd.read_csv('02_rmse_200.csv').values.flatten()

    # # random x - NN with x, y
    # rmse_data_random_xy450 = pd.read_csv('02_rmse_450.csv').values.flatten()

    rmse_data_random_xy_with_1000x = pd.read_csv('02_rmse_with_1000x.csv').values.flatten()

    # gridify
    rmse_data_gridify = pd.read_csv('../gridify/rmse/_ff1000_with_x/all_data.csv')['ext_normalized_rmse'].values

    # make figure
    plt.figure()
    rmse_data_fixed = rmse_data_fixed[~np.isnan(rmse_data_fixed)]
    rmse_data_fixed = rmse_data_fixed[~np.isinf(rmse_data_fixed)]

    rmse_data_random_y = rmse_data_random_y[~np.isnan(rmse_data_random_y)]
    rmse_data_random_y = rmse_data_random_y[~np.isinf(rmse_data_random_y)]

    rmse_data_random_xy_with_1000x = rmse_data_random_xy_with_1000x[~np.isnan(rmse_data_random_xy_with_1000x)]
    rmse_data_random_xy_with_1000x = rmse_data_random_xy_with_1000x[~np.isinf(rmse_data_random_xy_with_1000x)]

    rmse_data_gridify = rmse_data_gridify[~np.isnan(rmse_data_gridify)]
    rmse_data_gridify = rmse_data_gridify[~np.isinf(rmse_data_gridify)]

    plot_cdf(rmse_data_fixed, labels=False, ymax=1., color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$ (epochs=900,input=y)')
    plot_cdf(rmse_data_random_y, labels=False, ymax=1., color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=900,input=y)')
    plot_cdf(rmse_data_random_xy_with_1000x, labels=False, ymax=1., color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=1000,input=x,y)')
    # plot_cdf(rmse_data_random_xy450, labels=False, ymax=1., color='C5', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=450,input=x,y)')
    plot_cdf(rmse_data_gridify, labels=False, ymax=1., color='C7', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=900,gridify)')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig('03_eval_xy2eq-fixed-different_ymax1.pdf')

    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$ (epochs=900,input=y)')
    plot_cdf(rmse_data_random_y, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=900,input=y)')
    plot_cdf(rmse_data_random_xy_with_1000x, labels=False, color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=1000,input=x,y)')
    # plot_cdf(rmse_data_random_xy450, labels=False, color='C5', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=450,input=x,y)')
    plot_cdf(rmse_data_gridify, labels=False, color='C7', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (epochs=900,gridify)')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.ylim([0, 1000])
    plt.legend()
    plt.xscale('log')
    plt.savefig('03_eval_xy2eq-fixed-different_ymaxauto.pdf')

    # make figure x in [0, 3]
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.scale('linear')
    plt.legend()
    plt.savefig('03_eval_xy2eq-fixed-different.pdf')
