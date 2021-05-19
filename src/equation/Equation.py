"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: This class takes in a functional form and applies
         coefficients. Then, it can use y-values to determine
         the best coefficient values.

NOTES:

TODO:
"""
from srvgd.utils.rmse import RMSE
from gp.protected_functions import protected_exp, protected_log, pow2, pow3, pow4, pow5, pow6  # noqa: F401

from scipy.optimize import minimize
import numpy as np


class Equation:

    def __init__(self, eq_str: str,
                 x=np.arange(0.1, 3.1, 0.1),
                 num_coeffs: int = 0,
                 apply_coeffs: bool = True):

        self.eq_str = eq_str
        self.x = x

        if apply_coeffs:
            self.apply_coeffs()
        else:
            self.num_coeffs = 0

        self.get_f()

    def get_f(self):
        if hasattr(self, 'f'):
            return self.f

        if 'np.' not in self.eq_str:
            for p in ['sin', 'exp', 'log']:
                self.eq_str = self.eq_str.replace(p, 'np.'+p)

        if 'c[' in self.eq_str:
            lambda_str_beg = 'lambda c, x: '
        else:
            lambda_str_beg = 'lambda x: '

        try:
            self.f = eval(lambda_str_beg+self.eq_str)
            return self.f

        except SyntaxError:
            self.valid = False

    def is_valid(self):
        if hasattr(self, 'valid') and not self.valid:
            return False

        else:
            try:
                self.get_f()
                if self.num_coeffs == 0:
                    y = self.f(self.x)
                else:
                    y = self.f(np.ones(self.num_coeffs), self.x)
                return type(y) != np.ufunc
            except (SyntaxError, TypeError, AttributeError, NameError, FloatingPointError, ValueError):
                return False

    def apply_coeffs(self):
        pass

    def place_exact_coeffs(self, coeffs, remove_np=True):
        assert len(coeffs) == self.num_coeffs
        eq_placed_coeffs = self.eq_str
        for i, ci in enumerate(coeffs):
            eq_placed_coeffs = eq_placed_coeffs.replace('c[{}]'.format(i), str(ci))
        if remove_np:
            eq_placed_coeffs = eq_placed_coeffs.replace('np.', '')
        return eq_placed_coeffs

    def regression(self, y, init_guess_type='random',
                   init_guess=None):
        assert init_guess_type in ('random', 'ones', None)

        def loss(c, x):
            y_hat = self.f(c, x).flatten()
            return RMSE(y_hat=y_hat, y=y)

        if init_guess_type == 'random':
            init_guess = np.random.uniform(-3, 3, self.num_coeffs)
        elif init_guess_type == 'ones':
            init_guess = np.ones(self.num_coeffs)
        else:
            assert init_guess is not None

        res = minimize(loss, init_guess, args=(self.x,),
                       bounds=[(-3, 3)]*self.num_coeffs,
                       method='L-BFGS-B')

        rmse = loss(res.x, self.x)
        coeffs = res.x

        return coeffs, rmse

    def fit(self, y):
        assert y.ndim == 1
        assert self.x is not None, 'x not specified'
        assert self.num_coeffs > 0, 'num_coeffs must be > 0'

        rmse_list = []
        coeffs_list = []
        for _ in range(150):
            coeffs, rmse = self.regression(y, init_guess_type='random')

            rmse_list.append(rmse)
            coeffs_list.append(coeffs)

        # coeffs, rmse = self.regression(y, init_guess_type='ones')

        # rmse_list.append(rmse)
        # coeffs_list.append(coeffs)

        if np.all(np.isnan(rmse_list)):
            self.coeffs = coeffs_list[0]
            self.rmse = float('inf')
        else:
            index = np.nanargmin(rmse_list)
            self.coeffs = coeffs_list[index]
            self.rmse = rmse_list[index]

        return self.coeffs, self.rmse
