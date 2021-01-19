"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Given a list of functional forms, create a dataset
         of y-values (inputs) and functional forms (outputs).
         When using the generate_datasets.py there are many
         functional forms that are used only once. I don't
         think this should happen.

NOTES: ff = function form

TODO:
"""
from srvgd.utils.normalize import normalize
import numpy as np
import pandas as pd

import re


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
    coeff = get_coeff(rand_interval)
    ff_coeff += '+{:.3f}'.format(coeff)
    return ff_coeff, coeff_index+1


def get_dataset(support, ff_list, dataset_size):
    dataset_inputs = []
    dataset_outputs = []
    eq_with_coeff_list = []
    while len(dataset_inputs) < dataset_size:
        for ff in ff_list:
            ff_coeff = apply_coeffs(ff, (-3, 3))[0]
            f = eval('lambda x:'+numpify(ff_coeff))
            dataset_inputs.append(normalize(f(support)))
            dataset_outputs.append(ff)
            eq_with_coeff_list.append(ff_coeff)

    for i in range(10):
        print(i)
        print(dataset_inputs[i])
        print(dataset_outputs[i])
        print(eq_with_coeff_list[i])
        print()
        import matplotlib.pyplot as plt
        plt.close('all')
        plt.plot(support, dataset_inputs[i], '.-')
        plt.title(eq_with_coeff_list[i])
        plt.show()



def numpify(eq):
    for prim in ['sin', 'log', 'exp']:
        eq = eq.replace(prim, 'np.'+prim)
    return eq

if __name__ == '__main__':
    np.random.seed(0)

    # load functional forms
    ff_list = ['x', 'sin(x)']
    support = np.arange(0.1, 3.1, 0.1)
    dataset = get_dataset(support, ff_list, dataset_size=10)
