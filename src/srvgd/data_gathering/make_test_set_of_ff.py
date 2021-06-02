"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Get functional forms from unique_ff_list.csv
         and pick 1000 functional forms that are not
         contained in dataset_train_ff10000.pt

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import pandas as pd
import numpy as np

import os


if __name__ == '__main__':
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    full_ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten().tolist()

    dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                         map_location=torch.device('cpu'))

    train_ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]
    train_ff_list = np.unique(train_ff_list)

    non_train_ff_list = [ff for ff in full_ff_list if ff not in train_ff_list]

    np.random.seed(123)
    test_ff_list = np.random.choice(non_train_ff_list, replace=False, size=1000)
    pd.DataFrame(test_ff_list).to_csv(os.path.join(dataset_path, 'test_ff_list.csv'),
                                      index=False,
                                      header=None)
