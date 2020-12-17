"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 16, 2020

PURPOSE: Generate a dataset like the one described in

         Biggio, Luca, Tommaso Bendinelli, Aurelien Lucchi,
         and Giambattista Parascandolo. "A Seq2Seq approach
         to Symbolic Regression."

NOTES: Uses their code:
       https://github.com/SymposiumOrganization/EQLearner

TODO:
"""
from srvgd.updated_eqlearner.datasetcreator_rg import DatasetCreatorRG

import sympy
import numpy as np

import os


def save(path, dataset, info, eq_with_coeff, dataset_type):
    assert dataset_type in ('train', 'test')
    if dataset_type == 'train':
        assert info['isTraining']
    else:
        assert not info['isTraining']
    store_format = np.array((dataset, info), dtype='object')
    np.save(path+'_{}'.format(dataset_type), store_format)
    np.save(path+'_{}_with_coeffs'.format(dataset_type), eq_with_coeff)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple_scaling', action='store_true',
                        help='Scale each y-value with all y-values that '
                             'share the same x-value. Thus, y-values '
                             'at different x-values may scale differently.'
                             'If both multiple_scaling and consistent_scaling '
                             'are false, no scaling is performed.')
    parser.add_argument('--consistent_scaling', action='store_true',
                        help='Scale each y-value with the same scale. If both '
                             'multiple_scaling and consistent_scaling are false, '
                             'no scaling is performed.')

    args = parser.parse_args()
    assert not (args.multiple_scaling and args.consistent_scaling), 'cannot use --multiple_scaling and --consistent_scaling simultaneously'

    # get dataset_name from args
    if args.multiple_scaling:
        dataset_name = 'multiple_scaling'
    elif args.consistent_scaling:
        dataset_name = 'consistent_scaling'
    else:
        dataset_name = 'no_scaling'

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
    train_data = fun_generator.generate_set(support, 50000,
                                            isTraining=True,
                                            consistent_scaling=args.consistent_scaling,
                                            multiple_scaling=args.multiple_scaling)
    train_dataset, info_training, train_eq_with_coeff = train_data
    print('done.')

    print('Generating test dataset ... ', end='', flush=True)
    test_data = fun_generator.generate_set(support, 1000,
                                           isTraining=False,
                                           consistent_scaling=args.consistent_scaling,
                                           multiple_scaling=args.multiple_scaling)
    test_dataset, info_testing, test_eq_with_coeff = test_data
    print('done.')

    print('Saving datasets ... ', end='', flush=True)
    path = os.path.join('..', '..', '..', 'datasets', 'dataset_'+dataset_name)
    save(path, train_dataset, info_training, train_eq_with_coeff, 'train')
    save(path, test_dataset, info_testing, test_eq_with_coeff, 'test')
    print('done.')
