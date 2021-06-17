"""
AUTHOR: Ryan Grindle

LAST MODIFIED: June 15, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES:

TODO:
"""
from srvgd.utils.rmse import RMSE
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
Y = np.squeeze([d[0].tolist() for d in dataset])
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]
print(Y.shape)

min_list = pd.read_csv('min_list.csv', header=None).values.flatten()

ind_list = pd.read_csv('ind_list.csv', header=None).values.flatten()
assert len(ind_list) == len(min_list)

print('max error', max(min_list))

x = np.arange(0.1, 3.1, 0.1)

same_ff_count = 0
for count, (ind, error) in enumerate(zip(ind_list, min_list)):
    i, j = eval(ind)
    # print(i, RMSE(Y[(i+1) % 50000], Y[j]) - error <= 10**(-20))
    index_i = (i+1) % 50000
    Yi = Y[index_i]
    Yj = Y[j]
    assert RMSE(Yi, Yj) - error <= 10**(-20)

    if ff_list[index_i] == ff_list[j]:
        same_ff_count += 1

    plt.close('all')
    plt.figure()
    plt.plot(x, Yi, '.-', label=ff_list[index_i], alpha=0.5)
    plt.plot(x, Yj, '.-', label=ff_list[j], alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('RMSE = {:.3e}'.format(error))
    plt.legend()
    plt.savefig('similar_eq_in_dataset_{}.pdf'.format(count))

print('same_ff_count', same_ff_count)
