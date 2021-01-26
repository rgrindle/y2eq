"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 24, 2021

PURPOSE: Take a list of functional forms and generate
         a dataset of a given size from them. Inputs will
         be y-values and outputs will be the functional forms.

NOTES: Should the number of instances of each functional form
       in the dataset be related to the number of possible
       coefficients in the functional form?
       For now, each functional form will have about the
       same number of observations in the dataset.
       ff = functional form

TODO:
"""
from generate_dataset import pad
from tokenization_rg import tokenize_eq
from srvgd.utils.normalize import normalize

import torch
import numpy as np
import pandas as pd

import re

if torch.cuda.is_available():
    from tensor_dataset import TensorDatasetGPU as TensorDataset  # noqa: F401
else:
    from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401


def get_coeff(rand_interval):
    return np.around(np.random.uniform(*rand_interval), 3)


def apply_coeffs(ff, rand_interval=(-1, 1)):
    """Take functional form and put constants
    where applicable.

    Parameters
    ----------
    ff : str
        The functional form.
    rand_interval : tuple of length 2
        A list with the min and max numbers desired
        for coefficients.

    Returns
    -------
    ff_coeff : str
        equation str including adjustable coefficients.
    num_coeff : int
        The number of coefficients placed.

    Examples
    --------
    >>> np.random.seed(0)
    >>> apply_coeffs('x')
    ('0.098*x+0.430', 2)

    >>> apply_coeffs('sin(x)')
    ('0.090*sin(0.206*x)+-0.153', 3)

    >>> apply_coeffs('sin(exp(x))')
    ('-0.125*sin(0.784*exp(0.292*x))+0.927', 4)
    """
    assert len(rand_interval) == 2
    assert np.all(np.abs(rand_interval) < 10)

    coeff_index = 0

    # First, attach a coefficient to every occurance of x.
    # Be careful to find the variable not the x in exp, for example.
    eq_str_list = []
    for i, char in enumerate(ff):
        if char == 'x' and (i == 0 or ff[i-1] != 'e'):
            coeff = get_coeff(rand_interval)
            eq_str_list.append('{:.3f}*x'.format(coeff))
            coeff_index += 1
        else:
            eq_str_list.append(char)

    # Put a coefficient in front of every term.
    ff_coeff = ''.join(eq_str_list)
    ff_coeff_str_list = []
    for term in ff_coeff.split('+'):
        decimal_point_index = 1
        if term[0] == '-':
            decimal_point_index += 1

        if term[decimal_point_index+1:5].isdigit():
            ff_coeff_str_list.append(term)
        else:
            coeff = get_coeff(rand_interval)
            ff_coeff_str_list.append('{:.3f}*'.format(coeff)+term)
            coeff_index += 1
    ff_coeff = '+'.join(ff_coeff_str_list)

    # Put a coeff in front of any missed primitives.
    # Without this block sin(sin(x)) -> c[1]*sin(sin(c[0]*x))
    # but with this block sin(sin(x)) -> c[1]*sin(c[2]*sin(c[0]*x))
    for prim in ['sin', 'exp', 'log']:
        ff_coeff_str_list = []
        prev_i = 0
        for m in re.finditer(prim, ff_coeff):
            i = m.start()
            ff_coeff_str_list.append(ff_coeff[prev_i:i])
            if not ff_coeff[i-4:i-1].isdigit():
                coeff = get_coeff(rand_interval)
                ff_coeff_str_list.append('{:.3f}*'.format(coeff))
                coeff_index += 1
            prev_i = i
        ff_coeff_str_list.append(ff_coeff[prev_i:])
        ff_coeff = ''.join(ff_coeff_str_list)

    # Add verticle shift
    # coeff = get_coeff(rand_interval)
    # ff_coeff += '+{:.3f}'.format(coeff)
    return ff_coeff, coeff_index


def get_dataset(support, ff_list, dataset_size,
                support_ext,
                other_dataset_inputs=None):
    if other_dataset_inputs is None:
        other_dataset_inputs = []

    ff_list = list(ff_list)

    dataset_inputs = []
    dataset_outputs = []
    eq_with_coeff_list = []
    count = 0
    attempt_count = 0
    while count < dataset_size:
        ff = ff_list[count % len(ff_list)]
        ff_coeff = apply_coeffs(ff, (-3, 3))[0]
        f = eval('lambda x:'+numpify(ff_coeff))
        Y = f(support)
        attempt_count += 1
        if not np.any(np.isnan(Y)) and not np.any(np.isnan(f(support_ext))):
            if np.min(Y) != np.max(Y):
                if np.all(np.abs(Y) <= 1000):
                    normalized_Y = np.around(normalize(Y), 7).tolist()
                    if normalized_Y not in dataset_inputs and normalized_Y not in other_dataset_inputs:
                        dataset_inputs.append(normalized_Y)
                        tokenized_ff = tokenize_eq(ff)
                        dataset_outputs.append(torch.Tensor(tokenized_ff))
                        eq_with_coeff_list.append(ff_coeff)

                        print('.', flush=True, end='')
                        count += 1
                        attempt_count = 0

        if attempt_count > 100:
            del ff_list[count % len(ff_list)]
            attempt_count = 0

    print()
    return dataset_inputs, dataset_outputs, eq_with_coeff_list


def numpify(eq):
    for prim in ['sin', 'log', 'exp']:
        eq = eq.replace(prim, 'np.'+prim)
    return eq


def save(dataset_inputs, dataset_outputs, eq_with_coeff_list, save_name):
    inputs_tensor = torch.zeros(len(dataset_inputs), 30)
    for i, y in enumerate(dataset_inputs):
        inputs_tensor[i, :] = torch.Tensor(y)

    filename = 'equations_with_coeff'+save_name+'.csv'
    pd.DataFrame(eq_with_coeff_list).to_csv(filename, index=False, header=None)

    dataset = TensorDataset(inputs_tensor, pad(dataset_outputs))
    torch.save(dataset, 'dataset'+save_name+'.pt')


if __name__ == '__main__':
    np.random.seed(0)

    ff_list = pd.read_csv('unique_ff_list.csv', header=None).values.flatten()
    np.random.shuffle(ff_list)
    support = np.arange(0.1, 3.1, 0.1)

    dataset_parts = (None,)
    dataset_size = {'train': 50000, 'test': 1000}
    for dataset_type in ['train', 'test']:
        dataset_parts = get_dataset(ff_list=ff_list,
                                    support=support,
                                    support_ext=np.arange(3.1, 6.1, 0.1),
                                    dataset_size=dataset_size[dataset_type],
                                    other_dataset_inputs=dataset_parts[0])

        save(*dataset_parts, save_name='_'+dataset_type+'_ff')
