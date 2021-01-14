from cdf import plot_cdf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    rmse_data = pd.read_csv('rmse_test.csv').values[:, 1]
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    print(rmse_data[mask])
    plot_cdf(rmse_data[mask], labels=False, color='#8B94FC')
    plt.xlabel('RMSE')
    plt.ylabel('Cumulative Counts')
    # plt.xlim([0, 3])
    # plt.ylim([0, 1000])
    plt.savefig('figure_recreation2.pdf')
