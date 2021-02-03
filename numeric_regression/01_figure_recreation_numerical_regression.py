"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 14, 2021

PURPOSE: Take data from last script and plot
         a CDF to see if it matches paper.

NOTES:

TODO:
"""
from cdf import plot_cdf
import pandas as pd
import matplotlib.pyplot as plt

import os

df = pd.read_csv(os.path.join('rmse', 'all_data.csv'))

for key in df:
    plt.close('all')
    plt.figure()
    data = [v for v in df[key] if 0 <= v <= 10]
    plot_cdf(data, color='#CF6875')
    plt.ylabel('Cumulative Counts')
    plt.xlabel('RMSE on {}'.format(key))
    plt.xlim([0, 3])
    plt.ylim([0, 1000])
    plt.savefig('figure_recreation_numerical_regression_{}.pdf'.format(key))
