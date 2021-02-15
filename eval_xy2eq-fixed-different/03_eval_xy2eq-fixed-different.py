
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    rmse_xy2eq_fixed_different = pd.read_csv('02_rmse.csv', header=None).values.flatten()
    print(rmse_xy2eq_fixed_different.shape)

    rmse_xy2eq_fixed_different = rmse_xy2eq_fixed_different[~np.isnan(rmse_xy2eq_fixed_different)]
    rmse_xy2eq_fixed_different = rmse_xy2eq_fixed_different[~np.isinf(rmse_xy2eq_fixed_different)]

    # make figure
    plt.figure()
    plot_cdf(rmse_xy2eq_fixed_different, labels=False, ymax=1., label='xy2eq_fixed_different')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.savefig('03_eval_xy2eq-fixed-different_ymax1.pdf')

    plt.figure()
    plot_cdf(rmse_xy2eq_fixed_different, labels=False, label='xy2eq_fixed_different')
    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.ylim([0, 1000])
    plt.legend()
    plt.xscale('log')
    plt.savefig('03_eval_xy2eq-fixed-different_ymaxauto.pdf')

    # make figure x in [0, 3]
    plt.xscale('linear')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('03_eval_xy2eq-fixed-different.pdf')
