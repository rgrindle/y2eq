"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 15, 2021

PURPOSE: Get a big list of unique functional forms

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.DatasetCreatorRG import DatasetCreatorRG
from modifying_functional_forms_list import fix_ff

import numpy as np
import pandas as pd
from sympy import sin, log, exp, Symbol

SEED = 1234
np.random.seed(SEED)


if __name__ == '__main__':

    x = Symbol('x', real=True)
    basis_functions = [x, sin, log, exp]
    DC = DatasetCreatorRG(basis_functions,
                          max_linear_terms=1,
                          max_binomial_terms=1,
                          max_compositions=1,
                          max_N_terms=0,
                          division_on=False,
                          random_terms=True,
                          constants_enabled=False)

    ff_list = []
    num_ff = 0
    while num_ff < 51000:
        ff, _, _ = DC.generate_fun()

        ff_str = str(ff).replace(' ', '')

        try:
            ff_str = fix_ff(ff_str)
        except Exception as e:
            if 'This is a situation not concidered' in str(e):
                continue
            else:
                print('Unexpected exception caught:', str(e))

        if ff_str == '0':
            continue

        elif ff not in ff_list:
            ff_list.append(ff_str)
            num_ff += 1
            print(num_ff)

    pd.DataFrame(ff_list).to_csv('get_functional_forms.csv',
                                 header=None,
                                 index=False)
