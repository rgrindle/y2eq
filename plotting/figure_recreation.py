from srvgd.plotting.cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    import os

    # numerical regression data
    path = os.path.join('..', 'numeric_regression', 'rmse', '_ff1000')
    numerical_reg_data = pd.read_csv(os.path.join(path, 'all_data.csv'))['test_rmse'].values
    plot_cdf(numerical_reg_data, color='#CF6875', alpha=0.75, label='numeric regression')

    # gp data
    file = os.path.join('..', 'src', 'gp', 'gp_data', 'ff1000_100', 'all_rmse_data.csv')
    gp_data = pd.read_csv(file)['test_normalized_rmse'].values
    plot_cdf(gp_data, color='C8', alpha=0.75, label='genetic programming')

    # symbolic regression NN data
    file_endname = '_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900'
    sr_nn_data = pd.read_csv('../jupyter/02_rmse{}.csv'.format(file_endname)).values[:, 2]
    sr_nn_data = sr_nn_data[~np.isnan(sr_nn_data)]
    sr_nn_data = sr_nn_data[~np.isinf(sr_nn_data)]
    plot_cdf(sr_nn_data, labels=False, color='#8B94FC', alpha=0.75, label='symbolic regression NN')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.ylim([0, 1000])
    plt.legend()
    plt.xscale('log')
    plt.savefig('figure_recreation{}.pdf'.format(file_endname))

    plt.figure()

    # numeric
    numerical_reg_data = [v for v in numerical_reg_data if 0 <= v <= 3]
    plot_cdf(numerical_reg_data, color='#CF6875', alpha=0.75, label='numeric regression')

    # gp
    gp_data = [v for v in gp_data if 0 <= v <= 3]
    plot_cdf(gp_data, color='C8', alpha=0.75, label='genetic programming')

    # symbolic regression NN
    print(len(sr_nn_data))
    sr_nn_data = [r for r in sr_nn_data if 0 <= r <= 3]
    print(len(sr_nn_data))
    plot_cdf(sr_nn_data, labels=False, color='#8B94FC', alpha=0.75, label='symbolic regression NN')

    plt.xlabel('RMSE on extrapolation region ($x = \\left[3.1, 3.2, \\cdots, 6.0\\right]$)')
    plt.ylabel('Cumulative counts')
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.legend()
    plt.savefig('figure_recreation{}_0-3.pdf'.format(file_endname))
