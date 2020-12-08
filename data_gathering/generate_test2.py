"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 8, 2020

PURPOSE: Create the second test dataset. This dataset
         uses only equations containing primitive not
         used to created the training dataset. These
         primitives are cos, cosh, and /.

NOTES:

TODO:
"""
from DatasetGenerator import DatasetGenerator

import numpy as np

import os


MAX_DEPTH = 3
SEED = 1
unique_eqs_save_file = 'unique_eqs_maxdepth{}_test2.json'.format(MAX_DEPTH)
dataset_save_file = 'dataset_maxdepth{}_seed{}_test2.json'.format(MAX_DEPTH, SEED)
save_path = os.path.join('..', 'datasets')

DG = DatasetGenerator(num_args={'*': 2, '+': 2, 'cos': 1,
                                'cosh': 1, '/': 2},
                      max_depth=MAX_DEPTH,
                      X=np.linspace(0.1, 3.1, 30),
                      rng=np.random.RandomState(SEED),
                      include_zero_eq=True,
                      num_coeff_sets=1)

print('Removing equation without unknown primitives ...')
DG.all_eqs = [eq for eq in DG.all_eqs if any([p in eq.eq for p in ('cos', 'cosh', '/')])]

print('Saving unique equations ...')
DG.save_eqs(unique_eqs_save_file)
print('Unique equations saved:', unique_eqs_save_file)

print('Formatting unique equations into dataset ...')
DG.get_dataset()
print('Dataset has', len(DG.dataset_eqs), 'observations')

print('Saving dataset ...')
DG.save_dataset(dataset_save_file)
print('Dataset saved:', dataset_save_file)

if len(DG.non_dataset_eqs) > 0:
    print('Equations excluded from datatset '
          'because suitable constants could not '
          'be found are listed below.')
    for eq in DG.non_dataset_eqs:
        print(eq)
