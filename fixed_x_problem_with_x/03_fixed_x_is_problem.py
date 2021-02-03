
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

    # random x - NN with x, y
    rmse_data_random_xy = pd.read_csv('02_rmse.csv').values.flatten()

    # make figure
    plt.figure()
    rmse_data_fixed = rmse_data_fixed[~np.isnan(rmse_data_fixed)]
    rmse_data_fixed = rmse_data_fixed[~np.isinf(rmse_data_fixed)]

    rmse_data_random_y = rmse_data_random_y[~np.isnan(rmse_data_random_y)]
    rmse_data_random_y = rmse_data_random_y[~np.isinf(rmse_data_random_y)]

    rmse_data_random_xy = rmse_data_random_xy[~np.isnan(rmse_data_random_xy)]
    rmse_data_random_xy = rmse_data_random_xy[~np.isinf(rmse_data_random_xy)]

    plot_cdf(rmse_data_fixed, labels=False, ymax=1., color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random_y, labels=False, ymax=1., color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=y)')
    plot_cdf(rmse_data_random_xy, labels=False, ymax=1., color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=x,y)')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig('03_figure_fixed_x_is_problem_ymax1.pdf')

    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random_y, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=y)')
    plot_cdf(rmse_data_random_xy, labels=False, color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=x,y)')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.ylim([0, 1000])
    plt.legend()
    plt.xscale('log')
    plt.savefig('03_figure_fixed_x_is_problem_ymaxauto.pdf')

    # fixed x
    print(len(rmse_data_fixed))
    rmse_data_fixed = [r for r in rmse_data_fixed if 0 <= r <= 3]
    print(len(rmse_data_fixed))
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')

    # random x
    print(len(rmse_data_random_y))
    rmse_data_random_y = [r for r in rmse_data_random_y if 0 <= r <= 3]
    print(len(rmse_data_random_y))
    plot_cdf(rmse_data_random_xy, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=y)')

    print(len(rmse_data_random_xy))
    rmse_data_random_xy = [r for r in rmse_data_random_xy if 0 <= r <= 3]
    print(len(rmse_data_random_xy))
    plot_cdf(rmse_data_random_xy, labels=False, color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=x,y)')

    # make figure x in [0, 3]
    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random_y, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=y)')
    plot_cdf(rmse_data_random_xy, labels=False, color='C9', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$ (input=x,y)')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('03_figure_fixed_x_is_problem.pdf')
