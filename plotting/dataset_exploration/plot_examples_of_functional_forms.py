"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 10, 2021

PURPOSE: Examine the visual differences in equations
         that share the same functional form.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from equation.Equation import Equation
from srvgd.utils.normalize import normalize

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_examples(examples, x, ff, save_loc):
    plt.close('all')
    fig, axes = plt.subplots(ncols=2)
    for eq_str in examples:
        plt.sca(axes[0])
        eq = Equation(eq_str, apply_coeffs=False)
        y = eq.f(x)
        plt.plot(x, y, '.-')

        plt.sca(axes[1])
        plt.plot(x, normalize(y), '.-')

    plt.sca(axes[0])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(ff)

    plt.sca(axes[1])
    plt.xlabel('$x$')
    plt.ylabel('normalized $y$')
    plt.title(ff)

    plt.tight_layout()
    plt.savefig(save_loc)


if __name__ == '__main__':
    dataset_name = 'dataset_train_ff1000'

    dataset = torch.load('../../datasets/{}.pt'.format(dataset_name),
                         map_location=torch.device('cpu'))
    eq_list = pd.read_csv('../../datasets/equations_with_coeff{}.csv'.format(dataset_name[7:]), header=None).values.flatten()

    ff_list = np.array([get_eq_string(d[1].tolist())[5:-3] for d in dataset])
    unique_ff_list, inverse = np.unique(ff_list, return_inverse=True)

    ff2indices = {}
    ff2eqs = {}
    for i in range(len(unique_ff_list)):
        indices = np.where(inverse == i)[0]
        if len(indices) == 0:
            print(i)
        assert np.all(ff_list[indices] == ff_list[indices[0]])
        ff2indices[ff_list[indices[0]]] = indices
        ff2eqs[ff_list[indices[0]]] = eq_list[indices]

    x = np.arange(0.1, 3.1, 0.1)
    for ff_index, ff in enumerate(ff2eqs):
        print(ff_index, ff)
        plot_examples(examples=ff2eqs[ff],
                      x=x,
                      ff=ff,
                      save_loc='plot_examples_of_functional_forms/{}.pdf'.format(ff_index))
