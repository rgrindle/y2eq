"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: Run genetic programming.

NOTES:

TODO:
"""
from gp.GeneticProgramming import GeneticProgramming
from gp.RegressionDataset import RegressionDataset
from srvgd.utils.eval import get_f
from srvgd.utils.normalize import normalize, get_normalization_params

import numpy as np
import pandas as pd

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_index', type=int,
                    help='index in dataset to choose which equation to do.')
args = parser.parse_args()

np.random.seed(0)

# read in remaining datasets
eqs = pd.read_csv(os.path.join('..', '..', 'datasets', 'equations_with_coeff_test_ff1000.csv'), header=None).iloc[:, 0].values
print('shape of equation list', eqs.shape)

eq = eqs[args.dataset_index].replace('x', 'x[0]')
eq = eq.replace('ex[0]p', 'exp')  # fix messed up exp
print('target eq', eq)
f = get_f(eq)

x_int = RegressionDataset.linspace(0.1, 3.0, 30)
x_ext = RegressionDataset.linspace(3.1, 6.0, 30)

# split x_int into train (90%) and validation (10%)
i_val = np.random.choice(30, size=3, replace=False)
i_train = [i for i in range(30) if i not in i_val]
x_train = [x for i, x in enumerate(x_int) if i in i_train]
x_val = [x for i, x in enumerate(x_int) if i in i_val]

# Make the datasets
train_dataset = RegressionDataset(x=x_train, f=f)
val_dataset = RegressionDataset(x=x_val, f=f)
test_dataset = RegressionDataset(x=x_ext, f=f)

# import matplotlib.pyplot as plt
# train_dataset.plot()
# val_dataset.plot()
# test_dataset.plot()
# plt.show()
# exit()

gp = GeneticProgramming(exp=0,
                        rep=0,
                        pop_size=100,
                        max_gens=100,
                        dataset_index=args.dataset_index,
                        primitive_set=['*', '+', 'sin', 'log', 'exp',
                                       'pow2', 'pow3', 'pow4', 'pow5', 'pow6'],
                        terminal_set=['#x'],
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset)
gp.run('gp_data')

print(gp.best_individual.convert_lisp_to_standard())
print(gp.best_individual.coeffs)

train_rmse = gp.best_individual.fitness
test_rmse = gp.best_individual.testing_fitness

min_, scale_ = get_normalization_params(RegressionDataset.get_y(f=f, x=x_int).flatten())
train_normalized_rmse = normalize(train_rmse, min_=min_, scale_=scale_)
test_normalized_rmse = normalize(test_rmse, min_=min_, scale_=scale_)

os.makedirs('gp_data', exist_ok=True)

with open('gp_data/rmse{}.txt'.format(args.dataset_index), 'w') as f:
    f.write('train_normalized_rmse,train_rmse,test_normalized_rmse,test_rmse\n')
    f.write('{},{},{},{}'.format(train_normalized_rmse, train_rmse, test_normalized_rmse, test_rmse))
