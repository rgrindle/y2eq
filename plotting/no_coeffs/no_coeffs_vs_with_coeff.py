"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Compare y2eq and y2eq-no-coeff

NOTE:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.eval import remove_nan

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import os

model_name_list = ['y2eq-transformer-fixed-fixed',
                   'y2eq-transformer-no-coeffs-fixed-fixed',
                   'y2eq-transformer-fixed-fixed-new-ff',
                   'y2eq-transformer-no-coeffs-fixed-fixed-new-ff']
rmse_filename = {'y2eq-transformer-fixed-fixed': '02_rmse_150.csv',
                 'y2eq-transformer-no-coeffs-fixed-fixed': '02_rmse.csv',
                 'y2eq-transformer-fixed-fixed-new-ff': '02_rmse_150.csv',
                 'y2eq-transformer-no-coeffs-fixed-fixed-new-ff': '02_rmse.csv'}


def get_data(model_name):
    eval_loc = os.path.join('..', '..', 'eval_'+model_name)

    rmse = pd.read_csv(os.path.join(eval_loc, rmse_filename[model_name]))['rmse_int'].values
    print(len(rmse))

    return rmse


rmse_dict = {name: get_data(name) for name in model_name_list}
color = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}


plt.close('all')

fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for i, model_name in enumerate(rmse_dict):
    for ax in axes:
        plt.sca(ax)
        linestyle = 'solid'
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


result = mannwhitneyu(rmse_dict['y2eq-transformer-no-coeffs-fixed-fixed-new-ff'],
                      rmse_dict['y2eq-transformer-fixed-fixed-new-ff'],
                      alternative='greater')
print(result)
