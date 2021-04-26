"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 22, 2021

PURPOSE: Fit a functional form with coefficients many times.
         To do this for multiple truth values.

NOTES: Do this for one of the functional forms in
       dataset_test_ff1000.pt.

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from equation.EquationInfix import EquationInfix

import torch
import pandas as pd
import numpy as np

import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Fit coefficients to functional from 1000 times.')
parser.add_argument('--index', type=int, required=True,
                    help='Index in test dataset to use')
args = parser.parse_args()
print('index', args.index)

np.random.seed(0)

dataset_path = os.path.join('..', 'datasets')
test_dataset = torch.load(os.path.join(dataset_path, 'dataset_test_ff1000.pt'))
y_list = np.array([d[0].tolist() for d in test_dataset]).reshape((1000, 30))
ff_list = np.array([get_eq_string(d[1].tolist())[5:-3] for d in test_dataset])

eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_test_ff1000.csv'),
                      header=None).values.flatten()
print(eq_list.shape)
print(y_list.shape)
print(ff_list.shape)

print('ff', ff_list[args.index])
print('eq', eq_list[args.index])

eq = EquationInfix(ff_list[args.index],
                   x=np.arange(0.1, 3.1, 0.1))

rmse_list = []
for i in range(1000):
    coeffs, rmse = eq.regression(y_list[args.index], init_guess_type='random')
    print(i)
    print('coeffs', coeffs)
    print('rmse', rmse)
    rmse_list.append(rmse)
    print()

pd.DataFrame(rmse_list).to_csv('data/rmse_list_index{}.csv'.format(args.index),
                               header=None,
                               index=False)


rng_state = np.random.get_state()

with open('data/rng_state_index{}.pickle'.format(args.index), 'wb') as file:
    pickle.dump(rng_state, file)
