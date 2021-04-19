"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Get correct functional forms from the dataset.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string

import torch
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

dataset = torch.load('../datasets/dataset_test_ff1000.pt', map_location=device)
ff_list = [get_eq_string(d[1].tolist())[5:-3] for d in dataset]

pd.DataFrame(ff_list).to_csv('01_predicted_ff.csv', index=False, header=None)
