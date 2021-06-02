"""
AUTHOR: Ryan Grindle

LAST MOFIDIED: Jun 1, 2021

PURPOSE: Plot the data from random search and
         compare with y2eq.

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.eval import remove_nan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import os

eval_loc = os.path.join('..', 'eval_y2eq-transformer-fixed-fixed')

y2eq_rmse = pd.read_csv(os.path.join(eval_loc, '02_rmse_150.csv'))['rmse_int'].values
single_poly_rmse = pd.read_csv('rmse_single_poly.csv')['rmse_int'].values
single_sin_poly_rmse = pd.read_csv('rmse_single_sin_poly.csv')['rmse_int'].values
single_exp_poly_rmse = pd.read_csv('rmse_single_exp_poly.csv')['rmse_int'].values
single_complex_equation_rmse = pd.read_csv('rmse_single_complex_equation.csv')['rmse_int'].values

# single_log_poly_rmse = pd.read_csv('rmse_single_log_poly.csv')['rmse_int'].values
best_single = np.min([single_poly_rmse, single_sin_poly_rmse, single_exp_poly_rmse, single_complex_equation_rmse], axis=0)

rmse_dict = {'y2eq-transformer-fixed-fixed': y2eq_rmse,
             '$f_1(x) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1$': single_poly_rmse,
             '$f_2(x) = sin(x)^6 + sin(x)^5 + sin(x)^4 + sin(x)^3 + sin(x)^2 + sin(x) + 1$': single_sin_poly_rmse,
             '$f_3(x) = exp(x)^3 + exp(x)^2 + exp(x) + 1$': single_exp_poly_rmse,
             '$f_4(x) = sin(x)^6 + sin(x)^5 + sin(x)^4 + sin(x)^3 + sin(x)^2 + sin(x) + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1$': single_complex_equation_rmse,
             # '$log(x)^6 + log(x)^5 + log(x)^4 + log(x)^3 + log(x)^2 + log(x) + 1$': single_log_poly_rmse,
             '$\\min \\{f_1, f_2, f_3, f_4\\}$': best_single}
color = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5'}

plt.close('all')

fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for i, model_name in enumerate(rmse_dict):
    for ax in axes:
        plt.sca(ax)
        plot_cdf(remove_nan(rmse_dict[model_name]), labels=False, linestyle='solid', label=model_name, color=color[i])
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
plt.savefig('plot_y2eq_vs_random_search.pdf')

result = mannwhitneyu(rmse_dict['y2eq-transformer-fixed-fixed'],
                      rmse_dict['$\\min \\{f_1, f_2, f_3, f_4\\}$'],
                      alternative='less')
print(result)
