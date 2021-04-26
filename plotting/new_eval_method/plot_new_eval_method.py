"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 20, 2021

PURPOSE: Compare untrained model, trained model and optimizer

NOTES:

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.eval import remove_nan

import pandas as pd
import matplotlib.pyplot as plt

import os


model_name = 'y2eq-fixed-different'
model_type, _, eval_x_type = model_name.split('-')
untrained_name = model_type+'-untrained-'+eval_x_type
trained_eval_loc = os.path.join('..', '..', 'eval_'+model_name)
untrained_eval_loc = os.path.join('..', '..', 'eval_'+untrained_name)
bfgs_eval_loc = os.path.join('..', '..', 'eval_dataset-'+eval_x_type)

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


def plot_rmse_difference(rmse1, rmse2, name1, name2):
    data = sorted(remove_nan(rmse1 - rmse2), reverse=True)
    plt.plot(data, label=name1+' - '+name2)


plt.close('all')
plt.figure()
plot_rmse_difference(bfgs_rmse, old_rmse, 'optimizer RMSE', 'old RMSE')
plot_rmse_difference(bfgs_rmse, new_rmse, 'optimizer RMSE', 'new RMSE')
plt.yscale('symlog')
plt.ylabel('RMSE difference (linear scale near zero, log scale elsewhere)')
plt.xlabel('Equation (sorted by RMSE difference)')
plt.legend()
plt.show()

plt.close('all')
new_rmse = remove_nan(new_rmse)
old_rmse = remove_nan(old_rmse)
untrained_rmse = remove_nan(untrained_rmse)
bfgs_rmse = remove_nan(bfgs_rmse)

fig, axes = plt.subplots(ncols=2, sharey=True,
                         figsize=(2*6.4, 4.8))

for ax in axes:
    plt.sca(ax)
    plot_cdf(new_rmse, labels=False, label=model_name+' (new)')
    plot_cdf(old_rmse, labels=False, label=model_name+' (old)')
    plot_cdf(untrained_rmse, labels=False, label=untrained_name)
    plot_cdf(bfgs_rmse, labels=False, label='optimizer')

    plt.xlabel('RMSE on interpolation region ($x = \\left[0.1, 0.2, \\cdots, 3.0\\right]$)')

plt.sca(axes[0])
plt.xlim([0, 3])
plt.ylabel('Cummulative counts')

plt.sca(axes[1])
plt.xscale('log')
plt.legend()

plt.subplots_adjust(wspace=0.03, left=0.053, right=0.99, top=0.98)
plt.savefig(model_name+'.pdf')


# def plot_from_zero(y, color='C0', **kwargs):
#     for i, yi in enumerate(y):
#         plt.plot([i, i], [yi, 0], '.-',
#                  color=color, markevery=2,
#                  **kwargs)
