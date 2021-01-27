
from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # fixed x
    file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
    rmse_data = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]
    print(len(rmse_data))
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    rmse_data = [r for r in rmse_data[mask] if 0 <= r <= 3]
    print(len(rmse_data))
    plot_cdf(rmse_data, labels=False, color='#8B94FC', alpha=0.75, label='symbolic regression')

    # random x
    rmse_data = pd.read_csv('02_rmse.csv').values.flatten()
    print(len(rmse_data))
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    rmse_data = [r for r in rmse_data[mask] if 0 <= r <= 3]
    print(len(rmse_data))
    plot_cdf(rmse_data, labels=False, color='#F49F1C', alpha=0.75, label='symbolic regression')

    plt.xlabel('RMSE')
    plt.ylabel('Cumulative Counts')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('03_figure_fixed_x_is_problem.pdf')
