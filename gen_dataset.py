"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 10, 2020

PURPOSE: Generate a dataset like the one described in

         Biggio, Luca, Tommaso Bendinelli, Aurelien Lucchi,
         and Giambattista Parascandolo. "A Seq2Seq approach
         to Symbolic Regression."

NOTES: Uses their code:
       https://github.com/SymposiumOrganization/EQLearner

TODO:
"""
from datasetcreator_rg import DatasetCreatorRG

# from eqlearner.dataset.processing import tokenization
from eqlearner.dataset.utils import save_dataset

import sympy
import numpy as np

x = sympy.Symbol('x')
basis_functions = [x, sympy.sin, sympy.log, sympy.exp]

fun_generator = DatasetCreatorRG(basis_functions,
                                 max_linear_terms=1,
                                 max_binomial_terms=1,
                                 max_N_terms=1,
                                 max_compositions=2,
                                 constants_enabled=True)

support = np.linspace(0.1, 3.1, 30)
print('Generating train dataset ... ', end='', flush=True)
train_dataset, info_training = fun_generator.generate_set(support, 50000,
                                                          isTraining=True)
print('done.')

print('Generating test dataset ... ', end='', flush=True)
test_dataset, info_testing = fun_generator.generate_set(support, 1000,
                                                        isTraining=False)
print('done.')

print('Saving datasets ... ', end='', flush=True)
save_dataset(train_dataset, info_training,
             test_dataset, info_testing, path="./dataset.npy")
print('done.')
