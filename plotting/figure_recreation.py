from cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    import os

    # numerical regression data
    path = os.path.join('..', 'src', 'srvgd', 'numerical_regression', 'rmse')
    numerical_reg_data = pd.read_csv(os.path.join(path, 'all_data.csv'))['test_rmse'].values
    numerical_reg_data = [v for v in numerical_reg_data if 0 <= v <= 3]
    plot_cdf(numerical_reg_data, color='#CF6875', alpha=0.75, label='numeric regression NN')

    # symbolic regression data
    file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
    rmse_data = pd.read_csv('../jupyter/rmse{}.csv'.format(file_endname)).values[:, 1]
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    print(len(rmse_data))
    # print(rmse_data[mask])
    rmse_data = [r for r in rmse_data[mask] if 0 <= r <= 3]
    print(len(rmse_data))
    plot_cdf(rmse_data, labels=False, color='#8B94FC', alpha=0.75, label='symbolic regression NN (more valid)')

    # symbolic regression data other less valid
    file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
    rmse_data = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 1]
    print(len(rmse_data))
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    # print(rmse_data[mask])
    rmse_data = [r for r in rmse_data[mask] if 0 <= r <= 3]
    print(len(rmse_data))
    plot_cdf(rmse_data, labels=False, color='C2', alpha=0.75, label='symbolic regression NN (less valid)')

    plt.xlabel('RMSE')
    plt.ylabel('Cumulative Counts')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    # plt.title('Figure 1: RMSE')
    plt.savefig('figure_recreation{}.pdf'.format(file_endname))
