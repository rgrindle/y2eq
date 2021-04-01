"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Plot all the RMSE for the output ff's.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get RMSE from actual model
model_rmse = pd.read_csv('../../../eval_y2eq-fixed-fixed/02_rmse_105.csv')['rmse_int']
model_rmse = model_rmse[~np.isnan(model_rmse)]

rmse = pd.read_csv('rmse.csv')

for ff in rmse:
    plot_cdf(rmse[ff][~np.isnan(rmse[ff])],
             color='C0',
             alpha=0.5,
             label='unique ff\'s')

plot_cdf(model_rmse, color='C1', label='y2eq')

plt.xlabel('RMSE on interpolation region')
plt.ylabel('Cummulative counts')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xscale('log')

plt.savefig('plot_output_ff_rmse.pdf')
