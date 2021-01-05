"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 5, 2020

PURPOSE: See if coefficients can really be recalculated.

NOTES: We are dealing with scaled data.

TODO:
"""
from srvgd.utils.eval import fit_eq, get_f
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import numpy as np
import matplotlib.pyplot as plt
from eqlearner.dataset.processing.tokenization import get_string


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = torch.load('test_data_int_comp.pt', map_location=device)

x = np.arange(0.1, 3.1, 0.1)
for i, t in enumerate(test_data):
    y = np.array(t[0])
    eq = np.array(t[1])
    eq_str = get_string(eq)[5:-3]
    f = get_f(eq_str)
    coeff, rmse, fc = fit_eq([eq_str], x, [y])
    print(rmse)
    x_big = np.arange(0.1, 3.1, 0.001)
    plt.plot(x_big, fc[0](x_big), '-', label='recomputed coefficients')
    plt.plot(x, y, '.-', label='with coefficients')
    plt.title(eq_str)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('../figures/dataset_analysis/dataset_recomputed_consts_{}.jpg'.format(i))
    plt.close('all')
