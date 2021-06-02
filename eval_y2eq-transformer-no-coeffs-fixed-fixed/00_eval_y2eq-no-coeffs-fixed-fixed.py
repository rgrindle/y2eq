"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Get x and y to be used to evaluate y2eq-no-coeffs-fixed-fixed.

NOTES:

TODO:
"""
from srvgd.utils.eval import write_x_y_lists
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import numpy as np
import pandas as pd
import os


dataset = torch.load(os.path.join('..', 'datasets', 'dataset_train_ff1000_no_coeffs.pt'),
                     map_location=torch.device('cpu'))

dataset_output = [d[1].tolist() for d in dataset]
ff_list = [get_eq_string(out)[5:-3] for out in dataset_output]
print(len(ff_list))
ff_list = np.unique(ff_list)
print(len(ff_list))

pd.DataFrame(ff_list).to_csv('../datasets/equations_with_coeff_test_ff1000_no_coeffs.csv',
                             index=False,
                             header=None)

write_x_y_lists('../datasets/equations_with_coeff_test_ff1000_no_coeffs.csv',
                x_type='fixed')
