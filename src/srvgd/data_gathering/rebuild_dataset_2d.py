"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 12, 2021

PURPOSE: Take equation_with_coeffs_train_ff1000.csv and alter
         these equations to be two dimentional.

NOTES: For every variable x incountered, change it to
       x0 (with 50% change) or x1 (with a 50% chance)

TODO:
"""
import torch
import numpy as np
import pandas as pd

import os


def find_var_x(eq):
    indices = []
    for i, token in enumerate(eq):
        if token == 'x':
            if 0 < i < len(eq)-1:
                if eq[i-1:i+2] != 'exp':
                    indices.append(i)
            else:
                indices.append(i)
    return indices


def strip_x(eq):
    indices = find_var_x(eq)
    stripped_eq_list = []
    start = 0
    for i in indices:
        if start != i:
            stripped_eq_list.append(eq[start:i])
        start = i+1
    if start < len(eq):
        stripped_eq_list.append(eq[start:])
    return stripped_eq_list


def place_x(stripped_eq_list, x_choices=('x',)):

    # Determine if the final equation
    # will start or end with an x (or
    # neither start nor end with an x).
    x_len_offset = 0
    start_x, start_s = 1, 0
    if stripped_eq_list[0][0] in ('+', '*'):
        x_len_offset += 1
        start_x, start_s = 0, 1
    if stripped_eq_list[-1][-1] in ('+', '*'):
        x_len_offset += 1

    # assert len(stripped_eq_list)-1+x_len_offset == len(x_list)
    new_eq = [None]*(2*len(stripped_eq_list)-1+x_len_offset)
    new_eq[start_s::2] = stripped_eq_list
    new_eq[start_x::2] = np.random.choice(x_choices, size=len(stripped_eq_list)-1+x_len_offset,
                                          replace=True)
    return ''.join(new_eq)


def test_find_var_x():
    eq = 'sin(x)+exp(x)+x+x**2'
    indices = find_var_x(eq)

    print(eq)
    x_only_eq = ['x' if i in indices else ' ' for i, _ in enumerate(eq)]
    print(''.join(x_only_eq))

    assert len(indices) == 4


def test_strip_x():
    eq = 'x'
    stripped_eq_list = strip_x(eq)
    print(stripped_eq_list)
    assert stripped_eq_list == []

    eq = 'x**2+x'
    stripped_eq_list = strip_x(eq)
    print(stripped_eq_list)
    assert stripped_eq_list == ['**2+']

    eq = 'sin(x)+exp(x)+x+x**2'
    stripped_eq_list = strip_x(eq)
    print(stripped_eq_list)
    assert stripped_eq_list == ['sin(', ')+exp(', ')+', '+', '**2']

    eq = 'exp(x)'
    stripped_eq_list = strip_x(eq)
    print(stripped_eq_list)
    assert stripped_eq_list == ['exp(', ')']


def test_place_x():
    stripped_eq = ['sin(', ')']
    assert place_x(stripped_eq) == 'sin(x)'

    stripped_eq = ['+sin(', ')']
    assert place_x(stripped_eq) == 'x+sin(x)'

    stripped_eq = ['sin(', ')+']
    assert place_x(stripped_eq) == 'sin(x)+x'

    stripped_eq = ['*sin(', ')+']
    assert place_x(stripped_eq) == 'x*sin(x)+x'


test_find_var_x()
test_strip_x()
test_place_x()

if __name__ == '__main__':
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join('..', '..', '..', 'datasets')
    eq_list = pd.read_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000.csv'),
                          header=None).values.flatten()

    eq_2d_list = []
    for i, eq in enumerate(eq_list):
        stripped_eq_list = strip_x(eq)
        eq_2d = place_x(stripped_eq_list,
                        x_choices=('x[0]', 'x[1]'))
        eq_2d_list.append(eq_2d)

    pd.DataFrame(eq_2d_list).to_csv(os.path.join(dataset_path, 'equations_with_coeff_train_ff1000_2d.csv'),
                                    header=None,
                                    index=False)
