"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 18, 2021

PURPOSE: Get data from experiment 39 of MSR (meta symbolic regression)
         Will you this data to compare with y2eq.

NOTES: Don't forget 0 is removed (was index 5)

TODO:
"""
from srvgd.plotting.cdf import plot_cdf
from srvgd.utils.normalize import normalize
from srvgd.utils.rmse import RMSE
from PredictingUnseenFunctions.common.Kexpression import Kexpression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import os

path = '/Users/rgrindle/TlcsrSimplified_data/experiment39'

dataset = torch.load('../../datasets/dataset_msr.pt')
dataset_inputs = [np.array(d[0][:, 0].tolist()) for d in dataset]
# dataset_outputs = [d[1].tolist() for d in dataset]
print(len(dataset_inputs))

normalization_params_list = pd.read_csv('../../datasets/normalization_params_msr.csv', header=None).values

x = np.linspace(-1, 1, 20).reshape((1, 20))


def get_true_normalized_y(index):
    """Remember that originally index 5
    pointed to the 0 function. So, now
    if index is greater than 5 it should
    be reduced by 1. And if index is 5,
    return a bunch of zeros
    """
    if index < 5:
        return dataset_inputs[index]
    elif index > 5:
        return dataset_inputs[index-1]
    else:
        return np.zeros(20)


def get_pred_normalized_y(index, y_pred):
    if index < 5:
        normalization_params = normalization_params_list[index]
    elif index > 5:
        normalization_params = normalization_params_list[index-1]
    else:
        # basically don't normalize in this case
        normalization_params = [0., 1.]

    return normalize(y_pred, *normalization_params)


def get_rmse(rep):
    df_indices = pd.read_csv(os.path.join(path, 'rep{}_dataset_indices.csv'.format(rep)), header=None)

    indices = {}

    for dataset_type in ['train', 'validation', 'test']:
        indices[dataset_type] = df_indices.loc[df_indices[0] == dataset_type].values[0, 1:].astype(np.float64)
        not_nans = np.logical_not(np.isnan(indices[dataset_type]))
        indices[dataset_type] = indices[dataset_type][not_nans].astype(int)

    # for key in indices:
    #     print(len(indices[key]))

    df = pd.read_csv(os.path.join(path, 'rep{}_exp39_maxgenerations100_headlength5_popsize100_primitiveset*+-_terminalsetx0_split0.80.110.11_maxrewrites10_sigma0.5_latentdim8_rankfitnessFalse_useerrorestFalse_stdfitnesstypeadd_best_history.csv'.format(rep)))
    # columns = generation    dataset_type    dataset_index   best_error  best_kexpression

    max_gen = max(df['generation'])
    df_max_gen = df.loc[df['generation'] == max_gen]
    df_max_gen_test = df_max_gen.loc[df_max_gen['dataset_type'] == 'test']
    test_indices = indices['test']
    kexpressions = df_max_gen_test['best_kexpression'].values
    kexpressions = [eval(k) for k in kexpressions]
    kexpressions = [Kexpression(k, head_length=5) for k in kexpressions]
    functions = [k.get_function() for k in kexpressions]
    y = [f(x) for f in functions]
    normalized_y = []
    for i, j in enumerate(test_indices):
        normalized_y.append(get_pred_normalized_y(j, y[i]))

    true_normalized_y = [get_true_normalized_y(i) for i in test_indices]
    rmse_list = [RMSE(true, pred) for true, pred in zip(true_normalized_y, normalized_y)]
    return rmse_list


best_test_errors = []
for rep in range(9, 10):
    best_test_errors.append(get_rmse(rep))
    plot_cdf(best_test_errors[-1], labels=False)

best_test_errors_df = pd.DataFrame(best_test_errors)
print(best_test_errors_df)
best_test_errors_df.to_csv('mrs_data.csv', header=False, index=False)

plt.xlabel('RMSE')
plt.ylabel('Cummulative counts')
plt.show()
