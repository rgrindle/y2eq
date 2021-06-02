"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 18, 2021

PURPOSE: Compare y2eq and plot2eq with and without
         transformers.

NOTE:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.eval import remove_nan

import pandas as pd
import matplotlib.pyplot as plt

import os

model_name_list = ['plot2eq-fixed-fixed',
                   'plot2eq-transformer-fixed-fixed']


def get_data(model_name, filename='02_rmse_150.csv'):
    eval_loc = os.path.join('..', '..', 'eval_'+model_name)

    rmse = pd.read_csv(os.path.join(eval_loc, filename))['rmse_int'].values
    print(len(rmse))

    return rmse


rmse_dict = {name: get_data(name) for name in model_name_list}
color = {0: 'C0', 1: 'C1'}


plt.close('all')

fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for i, model_name in enumerate(rmse_dict):
    for ax in axes:
        plt.sca(ax)
        linestyle = 'solid'
        if i > 2:
            linestyle = 'dashed'
        plot_cdf(remove_nan(rmse_dict[model_name]), labels=False, linestyle=linestyle, label=model_name, color=color[i])
        plt.xlabel('RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)')

plt.sca(axes[0])
plt.xlim([0, 3])
plt.ylim([0, 1000])
plt.ylabel('Cummulative counts')

plt.sca(axes[1])
plt.xscale('log')
plt.ylim([0, 1000])
plt.legend()

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig('plot_'+model_name_list[0]+'_vs_'+model_name_list[1]+'.pdf')
