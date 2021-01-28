"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Update existing dataset to have triple the number of points
         We will see if training is easier on this dataset.

NOTES:

TODO:
"""
from srvgd.utils.eval import get_f
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import numpy as np
import pandas as pd

import os

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def redefine_dataset_change_x(end_name, support):
    eqs = pd.read_csv('equations_with_coeff{}.csv'.format(end_name),
                      header=None).values.flatten()

    dataset = torch.load(os.path.join('..', 'datasets', 'dataset{}.pt'.format(end_name)),
                         map_location=torch.device('cpu'))
    dataset_output_list = [d[1] for d in dataset]
    # func_forms = [get_string(d.tolist()) for d in dataset_output_list]

    dataset_inputs = torch.zeros(len(dataset), len(support))
    for i, _ in enumerate(eqs):
        f = get_f(eqs[i])
        y = f(support)
        if np.any(np.isnan(y)):
            dataset_inputs[i, :] = dataset_inputs[i-1, :]
        else:
            dataset_inputs[i, :] = torch.Tensor(y)
        print('', flush=True, end='')

    dataset_outputs = torch.zeros(len(dataset), len(dataset_output_list[0]))
    for i, o in enumerate(dataset_output_list):
        dataset_outputs[i, :] = o
    return TensorDataset(dataset_inputs, dataset_outputs.long())


if __name__ == '__main__':
    support = np.arange(0.1, 3.1, 1./30.)

    for dataset_type in ['train', 'test']:
        dataset_train = redefine_dataset_change_x(end_name='_'+dataset_type,
                                                  support=support)
        torch.save(dataset_train, 'dataset_{}_triple_points.pt'.format(dataset_type))
