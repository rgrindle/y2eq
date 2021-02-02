
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # fixed x
    file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
    rmse_data_fixed = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]

    # random x
    rmse_data_random = pd.read_csv('02_rmse_ff.csv').values.flatten()

    # make figure
    plt.figure()
    rmse_data_fixed = rmse_data_fixed[~np.isnan(rmse_data_fixed)]
    rmse_data_random = rmse_data_random[~np.isnan(rmse_data_random)]
    plot_cdf(rmse_data_fixed, labels=False, ymax=1., color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random, labels=False, ymax=1., color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig('03_figure_fixed_x_is_problem_ymax1.pdf')

    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')
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
    print(len(rmse_data_random))
    rmse_data_random = [r for r in rmse_data_random if 0 <= r <= 3]
    print(len(rmse_data_random))
    plot_cdf(rmse_data_random, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')

    # make figure x in [0, 3]
    plt.figure()
    plot_cdf(rmse_data_fixed, labels=False, color='#8B94FC', label='$x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$')
    plot_cdf(rmse_data_random, labels=False, color='#F49F1C', label='$x \\sim $uniform$\\left([0.1, 3.1)\\right)$')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('03_figure_fixed_x_is_problem.pdf')
