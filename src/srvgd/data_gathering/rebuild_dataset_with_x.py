"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 28, 2021

PURPOSE: Read in equations_with_coeff for a specified dataset
         then recompute y-values multiple times for each equations
         on different x-values.

NOTES: x in [0.1, 3.1)

TODO:
"""
from srvgd.updated_eqleaner.TensorDataset_rg import TensorDatasetCPU as TensorDataset
from srvgd.utils.eval import get_f, normalize

import torch
import pandas as pd
import numpy as np

import os


def rebuild_dataset_with_x(num_x_per_eq, eq_list, other_dataset_inputs):
    x_min_ = 0.1
    x_scale = 1./(3.0-0.1)

    dataset_inputs = []
    dataset_outputs = []
    for eq_index, eq in enumerate(eq_list):
        count = 0
        while count <= num_x_per_eq:
            f = get_f(eq)
            X = np.random.uniform(0.1, 3.1, 30)
            X.sort()
            Y = f(X)

            # Since I have checked this for the gridified x-values
            # when generating the dataset, I don't expect many (if any)
            # equations/x-value choices to be thrown out.
            # But, I'll check to be safe.
            if not np.any(np.isnan(Y)):
                if np.min(Y) != np.max(Y):
                    if np.all(np.abs(Y) <= 1000):
                        normalized_Y = np.around(normalize(Y), 7).tolist()
                        if normalized_Y not in dataset_inputs and normalized_Y not in other_dataset_inputs:
                            count += 1
                            normalized_X = normalize(X, min_=x_min_, scale=x_scale)
                            inp = np.vstack((normalized_X, normalized_Y))
                            print(inp.shape)
                            exit()
                            dataset_inputs.append(normalized_Y)
                            dataset_outputs.append(original_outputs[eq_index])

    return dataset_inputs, dataset_outputs


if __name__ == '__main__':
    np.random.seed(0)

    num_x_per_eq = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join('..', '..', '..', 'datasets')

    other_dataset_inputs = None
    for dataset_type in ['train', 'test']:
        dataset_name = '_test_ff'
        eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff'+dataset_name+'.csv'),
                              header=None).values.flatten()

        original_dataset = torch.load(os.path.join(dataset_path, 'dataset'+dataset_name+'.pt'),
                                      map_location=device)
        original_outputs = [d[1].tolist() for d in original_dataset]

        rebuild_dataset_with_x(num_x_per_eq, eq_list, other_dataset_inputs)
