"""
AUTHOR: Ryan Grindle

LAST MODIFIED: June 29, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES: Do the same thing as 02 but plot groups of plots
       together.

TODO:
"""
from srvgd.utils.rmse import RMSE
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def latex_it(eq_str):
    eq_str = eq_str.replace('**', '^')
    eq_str = eq_str.replace('*', '')
    eq_str = eq_str.replace('log', '\\log')
    eq_str = eq_str.replace('sin', '\\sin')
    eq_str = eq_str.replace('exp', '\\exp')

    return '$'+eq_str+'$'


dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
Y = np.squeeze([d[0].tolist() for d in dataset])
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]
print(Y.shape)

min_list = pd.read_csv('min_list.csv', header=None).values.flatten()

ind_list = pd.read_csv('ind_list.csv', header=None).values.flatten()
assert len(ind_list) == len(min_list)

print('max error', max(min_list))


def plot(nrows, count=0):
    assert type(nrows) == int
    assert nrows >= 1

    x = np.arange(0.1, 3.1, 0.1)

    fig, axes = plt.subplots(nrows=nrows, ncols=5,
                             figsize=(5, nrows),
                             sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.97,
                        bottom=0.08, top=0.99,
                        wspace=0.25, hspace=0.25)
    axes = axes.flatten()

    not_same_ff_count = 0
    for count, (ind, error) in enumerate(zip(ind_list[count:], min_list[count:])):
        i, j = eval(ind)
        # print(i, RMSE(Y[(i+1) % 50000], Y[j]) - error <= 10**(-20))
        index_i = (i+1) % 50000
        Yi = Y[index_i]
        Yj = Y[j]
        assert RMSE(Yi, Yj) - error <= 10**(-20)

        if ff_list[index_i] != ff_list[j]:
            plt.sca(axes[not_same_ff_count])

            plt.plot(x, Yi, '.-', label=latex_it(ff_list[index_i]),
                     alpha=0.5, markersize=2)
            plt.plot(x, Yj, '.-', label=latex_it(ff_list[j]),
                     alpha=0.5, markersize=1)

            if not_same_ff_count >= (nrows-1)*5:
                plt.xlabel('$x$')
            if (not_same_ff_count % 5) == 0:
                plt.ylabel('$y$')
            plt.text(0.5, 0.5, 'RMSE = {:.2e}'.format(error), fontsize=4)
            plt.legend(fontsize=2,
                       handlelength=0.5, markerscale=0.5)

            not_same_ff_count += 1

            if not_same_ff_count >= nrows*5:
                break

    return count+1


count = plot(nrows=8, count=0)
plt.savefig('similar_eq_in_dataset_grouped1.pdf')

plot(nrows=7, count=count)
plt.savefig('similar_eq_in_dataset_grouped2.pdf')
