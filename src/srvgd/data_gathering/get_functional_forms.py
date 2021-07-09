"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 15, 2021

PURPOSE: Get a big list of unique functional forms

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix
from srvgd.updated_eqlearner.DatasetCreatorRG import DatasetCreatorRG
from modifying_functional_forms_list import fix_ff

import numpy as np
import pandas as pd
from sympy import sin, log, exp, Symbol

SEED = 1234
np.random.seed(SEED)
x_int = np.arange(0.1, 3.1, 0.1)


def acceptible_ff(ff_str):
    f = EquationInfix(ff_str, apply_coeffs=False).f
    y = f(x_int)
    no_nans = np.all(np.logical_not(np.isnan(y)))
    y_values_in_range = np.max(np.abs(y)) <= 1000
    not_const = np.min(y) != np.max(y)
    return no_nans and y_values_in_range and not_const


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

        if 'E' in ff_str:
            # If ff_str contains E (sympy's number e)
            # then it breaks our rule that says ff's
            # do not contain any coefficients other than
            # 1. Fixing the functional form by removing E
            # is difficult in some cases e.g. E*x + x should
            # become x, but simply removing E* would result
            # in 2*x which contains another coefficient. There
            # are even more complex examples that include composite
            # functions...
            continue

        try:
            ff_str = fix_ff(ff_str)
        except Exception as e:
            if 'This is a situation not concidered' in str(e):
                continue
            else:
                print('ff_str =', ff_str)
                print('Unexpected exception caught:', str(e))
                exit()

        if ff_str == '0':
            continue

        elif ff_str not in ff_list:
            if acceptible_ff(ff_str):
                ff_list.append(ff_str)
                num_ff += 1
                print(num_ff)

        if num_ff > 1 and (num_ff % 1000) == 0:
            pd.DataFrame(ff_list).to_csv('get_functional_forms_week.csv',
                                         header=None,
                                         index=False)

    pd.DataFrame(ff_list).to_csv('get_functional_forms_week.csv',
                                 header=None,
                                 index=False)
