"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 20, 2021

PURPOSE: Compare untrained model, trained model and optimizer

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.eval import remove_nan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

model_name1 = 'y2eq-fixed-normal'
model_name2 = 'xy2eq-fixed-normal'


def get_data(model_name):
    model_type, _, eval_x_type = model_name.split('-')
    untrained_name = model_type+'-untrained-'+eval_x_type
    trained_eval_loc = os.path.join('..', '..', 'eval_'+model_name)
    untrained_eval_loc = os.path.join('..', '..', 'eval_'+untrained_name)
    bfgs_eval_loc = os.path.join('..', '..', 'eval_dataset-fixed')

    new_rmse = pd.read_csv(os.path.join(trained_eval_loc, '02_rmse.csv'))['rmse_int'].values
    print(len(new_rmse))
    # print(len(new_rmse))

    old_rmse = pd.read_csv(os.path.join(trained_eval_loc, '02_rmse_single_BFGS.csv'))['rmse_int'].values
    print(len(old_rmse))
    # print(len(old_rmse))

    untrained_rmse = pd.read_csv(os.path.join(untrained_eval_loc, '02_rmse.csv'))['rmse_int'].values
    print(len(untrained_rmse))
    # print(len(untrained_rmse))

    bfgs_rmse = pd.read_csv(os.path.join(bfgs_eval_loc, '02_rmse.csv'))['rmse_int'].values
    print(len(bfgs_rmse))
    # print(len(bfgs_rmse))

    return {'new': new_rmse,
            'old': old_rmse,
            'untrained': untrained_rmse,
            'bfgs': bfgs_rmse}


def plot_rmse_difference(rmse1, rmse2, name1, name2):
    data = sorted(remove_nan(rmse1 - rmse2), reverse=True)
    plt.plot(data, label=name1+' - '+name2)


rmse1_dict = get_data(model_name1)
rmse2_dict = get_data(model_name2)

plt.close('all')
plt.figure()
plot_cdf(remove_nan(rmse1_dict['bfgs']-rmse1_dict['new']), label='optimizer RMSE - '+model_name1+' RMSE')
plot_cdf(remove_nan(rmse2_dict['bfgs']-rmse2_dict['new']), label='optimizer RMSE - '+model_name2+' RMSE')
plt.xscale('symlog')
plt.xlabel('RMSE difference (linear scale near zero, log scale elsewhere)')
plt.ylabel('Cummulative counts')
plt.legend()
plt.savefig('diff_'+model_name1+'_vs_'+model_name2+'.pdf')


plt.close('all')
for key in rmse1_dict:
    rmse1_dict[key] = remove_nan(rmse1_dict[key])
    rmse2_dict[key] = remove_nan(rmse2_dict[key])

fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for i, key in enumerate(rmse1_dict):
    for ax in axes:
        plt.sca(ax)
        plot_cdf(rmse1_dict[key], labels=False, label=model_name1+' ('+key+')', color='C{}'.format(i))
        plot_cdf(rmse2_dict[key], labels=False, linestyle='dashed', label=model_name2+' ('+key+')', color='C{}'.format(i))

        plt.xlabel('RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)')

plt.sca(axes[0])
plt.xlim([0, 3])
plt.ylabel('Cummulative counts')

plt.sca(axes[1])
plt.xscale('log')
plt.legend()

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig(model_name1+'_vs_'+model_name2+'.pdf')


# def plot_from_zero(y, color='C0', **kwargs):
#     for i, yi in enumerate(y):
#         plt.plot([i, i], [yi, 0], '.-',
#                  color=color, markevery=2,
#                  **kwargs)
