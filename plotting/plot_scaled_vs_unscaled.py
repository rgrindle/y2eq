"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 18, 2020

PURPOSE: Explore different methods for scaling y-values
         (NN inputs).

NOTES:

TODO:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy
from sympy import sin, exp, log

import os

dataset_name = 'no_scaling_train'

path = os.path.join('..', 'datasets')
dataset, info = np.load(os.path.join(path, 'dataset_'+dataset_name+'.npy'), allow_pickle=True)
dataset_inputs = np.array([np.array(d[0]) for d in dataset])
dataset_outputs = np.array([np.array(d[1]) for d in dataset])
print(dataset_inputs.shape)
print(dataset_outputs.shape)

eq_with_coeff = pd.read_csv(os.path.join(path, 'dataset_'+dataset_name+'_with_coeffs.csv'), header=None).values
print(eq_with_coeff.shape)

support = np.linspace(0.1, 3.1, 30)

save_path = os.path.join('scaled_vs_unscaled', dataset_name)
os.makedirs(save_path, exist_ok=True)

x = sympy.symbols('x')
for i in range(100):
    plt.close('all')
    fig, axes = plt.subplots(ncols=2)
    print(eq_with_coeff[i, 0])
    sympy_expr = eval(eq_with_coeff[i, 0])
    if str(sympy_expr) == '0':
        f = lambda x: 0*x
    else:
        f = sympy.lambdify(x, sympy_expr)
    plt.sca(axes[0])
    plt.plot(support, dataset_inputs[i], label='scaled')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.sca(axes[1])
    plt.plot(support, f(support), 'C1', label='unscaled')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    fig.suptitle(eq_with_coeff[i, 0])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plot_scaled_vs_unscaled_{}_{}.png'.format(dataset_name, i)))
