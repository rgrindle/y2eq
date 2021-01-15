from cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
    rmse_data = pd.read_csv('../jupyter/rmse{}.csv'.format(file_endname)).values[:, 1]
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    print(rmse_data[mask])
    plot_cdf(rmse_data[mask], labels=False, color='#8B94FC')
    plt.xlabel('RMSE')
    plt.ylabel('Cumulative Counts')
    # plt.xlim([0, 3])
    # plt.ylim([0, 1000])
    plt.savefig('figure_recreation{}.pdf'.format(file_endname))
