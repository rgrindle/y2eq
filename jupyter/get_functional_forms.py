"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Update existing dataset to have triple the number of points
         We will see if training is easier on this dataset.

NOTES: Modified from SeqSeqModel.ipynb

TODO:
"""
from eqlearner.dataset.processing.tokenization import get_string
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import numpy as np

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_functional_forms(end_name):
    dataset = torch.load('dataset{}.pt'.format(end_name),
                         map_location=torch.device('cpu'))
    dataset_output_list = [d[1] for d in dataset]
    return [get_string(d.tolist()) for d in dataset_output_list]


if __name__ == '__main__':

    func_forms = get_functional_forms(end_name='_train')
    unique_func_forms = np.unique(func_forms)
    print(len(unique_func_forms))
    ff_counts = np.zeros_like(unique_func_forms, dtype=int)
    for i, ff1 in enumerate(unique_func_forms):
        for ff2 in func_forms:
            if ff1 == ff2:
                ff_counts[i] += 1
    print(ff_counts)
    assert np.all(ff_counts > 0)
    # import matplotlib.pyplot as plt
    # plt.plot(sorted(ff_counts))
    indices = np.argsort(ff_counts)

    for i in indices[::-1]:
        print(ff_counts[i], unique_func_forms[i])
