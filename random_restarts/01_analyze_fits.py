"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 26, 2021

PURPOSE: Determine an appropriate number of restarts to
         use with BFGS.

NOTES: Take the number of restarts for each run where 90%
       of the final RMSE was achieved.

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def forward_fill_nan(data):
    """If data contains NaN's
    replace with previous non-NaN."""
    prev_non_nan = np.nanmax(data)

    new_data = []
    for d in data:
        if np.isnan(d):
            new_data.append(prev_non_nan)
        else:
            new_data.append(d)

        prev_non_nan = new_data[-1]

    return new_data


def get_within_percent_final(data, frac):
    """Get the number of restarts it took
    to reach data[-1]*frac."""

    target = data[-1]+(1-frac)*(data[0]-data[-1])
    withhin_target_ind = np.where(data <= target)[0]
    return np.min(withhin_target_ind)


rmse_list_list = []
within_percent_indices = []
for index in range(1000):
    rmse_list = pd.read_csv('data/rmse_list_index{}.csv'.format(index), header=None).values.flatten()
    rmse_list = forward_fill_nan(rmse_list)
    rmse_list = np.minimum.accumulate(rmse_list)
    rmse_list_list.append(rmse_list)
    within_percent_index = get_within_percent_final(rmse_list, frac=.9)
    # print(within_percent_index)
    # plt.plot(rmse_list)
    # plt.vlines(within_percent_index, *plt.ylim())
    # hl = [(1-f)*(np.max(rmse_list)-np.min(rmse_list))+np.min(rmse_list) for f in np.arange(0, 1, 0.01)]
    # print(hl)
    # plt.hlines(hl, *plt.xlim())
    # plt.yscale('log')
    # plt.show()
    within_percent_indices.append(within_percent_index)


for within_percent_index in within_percent_indices:
    plt.plot([within_percent_index]*2, plt.ylim(), 'k', alpha=0.1)

for rmse_list in rmse_list_list:
    plt.plot(rmse_list, color='C0', alpha=0.1)

plt.yscale('log')
plt.ylabel('Root mean squared error')
plt.xlabel('Number of random restarts')
plt.show()
