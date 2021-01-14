"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 12, 2021

PURPOSE: PURPOSE: Generate train dataset of 50000 observations where
         each observation is a symbolic regression problem/answer.
         This time check if 30 points is enough. Reject equations
         for which 30 points is not enough.

NOTES: Modified from SeqSeqModel.ipynb

TODO:
"""
from DatasetCreatorRG import DatasetCreatorRG
from get_points import is_enough_points
from eqlearner.dataset.processing import tokenization

import torch
import numpy as np
import pandas as pd
from sympy import sin, log, exp, Symbol

import os

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# if torch.cuda.is_available():
#     from tensor_dataset import TensorDatasetGPU as TensorDataset  # noqa: F401
# else:
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401


def get_dataset(dataset_size, save_loc='',
                save_name='_test', other_dataset=None):
    """Create and save dataset. File will be
    saved in directory save_loc with name
    'dataset'+save_name+'.pt'

    The equations with coefficients will also
    be saved in save_loc
    with the name 'equations_with_coeff'+save_name+'.csv'

    other_dataset specifies a dataset that you want to
    have no shared observations with.
    """
    inputs, outputs, eq_list = get_dataset_data(dataset_size, other_dataset)
    outputs_padded = pad(outputs)

    dataset = TensorDataset(inputs, outputs_padded.long())

    save(dataset, eq_list, save_loc, save_name)
    return dataset


def get_dataset_data(dataset_size, other_dataset=None):
    """Generate unique equations and
    record some details. Equations that
    have y-values outsize of [-1000, 1000]
    are discarded.

    Parameters
    ----------
    dataset_size : int
        The number of unique equations to create.
        This is the same as the number of observations
        in the final dataset.
    other_dataset : TensorDataset (or None)
        A dataset that you don't want to share
        any observations with the dataset to be
        created.

    Returns
    -------
    dataset_input : list
        The y-values of each unique equation.
        dataset_input.shape = (dataset_size, 30)
    dataset_output : list (jagged)
        The tokenized equation without coefficients.
        len(dataset_output) = dataset_size. But equations
        are likely not the same size.
    eq_list : list
        A list of equations with coefficients.
        len(eq_list) = dataset_size. But equations
        are likely not the same size.
    """
    if other_dataset is None:
        other_dataset = []

    dataset_input = []
    dataset_output = []
    eq_list = []
    count = 0
    while count < dataset_size:
        eq, dictionary, dictionary_cleaned = DC.generate_fun()
        Y = DC.evaluate_function(support, eq, X_noise=False).tolist()
        if not np.any(np.isnan(Y)):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    f = eval('lambda x:'+str(eq).replace('sin', 'np.sin').replace('log', 'np.log').replace('exp', 'np.exp'))
                    if is_enough_points(f, support, 0.1):
                        normalized_Y = normalize(Y).tolist()
                        if normalized_Y not in dataset_input and normalized_Y not in other_dataset:
                            dataset_input.append(normalized_Y)
                            tokenized_eq = tokenization.pipeline([dictionary_cleaned])[0]
                            dataset_output.append(torch.Tensor(tokenized_eq))

                            eq_list.append(str(eq))

                            print('.', flush=True, end='')
                            count += 1

    print()
    inputs_tensor = torch.zeros(len(dataset_input), 30)
    for i, y in enumerate(dataset_input):
        inputs_tensor[i, :] = torch.Tensor(y)
    return dataset_input, dataset_output, eq_list


def pad(unpadded):
    """pad unpadded (tokenized equations) so they are
    all the same length.

    Parameters
    ----------
    unpadded : 2D list
        The data to be padded. Expected to be
        tokenized equations.

    Returns
    -------
    padded : 2D list
        The padded data. (len(unpadded) = len(padded))
    """
    max_len = np.max([len(y) for y in unpadded])
    padded = [np.hstack((eq_seq, np.zeros(max_len-len(eq_seq)))) for eq_seq in unpadded]
    return torch.Tensor(padded)


def save(dataset, eq_list,
         save_loc, save_name):
    if save_loc is None:
        return

    if save_loc != '':
        os.makedirs(save_loc, exist_ok=True)

    # save equations with coefficients
    filename = os.path.join(save_loc, 'equations_with_coeff'+save_name+'.csv')
    pd.DataFrame(eq_list).to_csv(filename, index=False, header=None)

    # save dataset
    filename = os.path.join(save_loc, 'dataset'+save_name+'.pt')
    torch.save(dataset, filename)


def normalize(data):
    _min = np.min(data)
    _max = np.max(data)
    return np.around((data-_min)/(_max-_min), 7)


if __name__ == '__main__':

    x = Symbol('x', real=True)
    basis_functions = [x, sin, log, exp]
    support = np.arange(0.1, 3.1, 0.1)
    DC = DatasetCreatorRG(basis_functions,
                          max_linear_terms=1,
                          max_binomial_terms=1,
                          max_compositions=1,
                          max_N_terms=0,
                          division_on=False,
                          random_terms=True,
                          constants_enabled=True,
                          constant_intervals_ext=[(-3, 1), (1, 3)],
                          constant_intervals_int=[(1, 3)])

    dataset_train = get_dataset(dataset_size=50000, save_name='_enough_train')
    dataset_test = get_dataset(dataset_size=1000, save_name='_enough_test',
                               other_dataset=dataset_train)
