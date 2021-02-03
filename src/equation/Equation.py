"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: This class takes in a functional form and applies
         coefficients. Then, it can use y-values to determine
         the best coefficient values.

NOTES:

TODO:
"""
from gp.protected_functions import protected_exp, protected_log, pow2, pow3, pow4, pow5, pow6  # noqa: F401

from scipy.optimize import minimize
import numpy as np


class Equation:

    def __init__(self, eq_str: str,
                 x=None,
                 num_coeffs: int = 0,
                 apply_coeffs: bool = True):

        self.eq_str = eq_str
        self.x = x

        if apply_coeffs:
            self.apply_coeffs()
        else:
            self.num_coeffs = num_coeffs

        self.get_f()

    def get_f(self):
        if 'c[' in self.eq_str:
            lambda_str_beg = 'lambda c, x:'
        else:
            lambda_str_beg = 'lambda x:'
        self.f = eval(lambda_str_beg+self.eq_str)

    def apply_coeffs(self):
        pass

    def fit(self, y):
        assert self.x is not None, 'x not specified'
        assert self.num_coeffs > 0, 'num_coeffs must be > 0'

        def loss(c, x):
            y_hat = self.f(c, x).flatten()
            return np.sqrt(np.mean(np.power(y-y_hat, 2)))

        res = minimize(loss, np.ones(self.num_coeffs),
                       args=(self.x,),
                       bounds=[(-3, 3)]*self.num_coeffs,
                       method='L-BFGS-B')

        self.coeffs = res.x
        self.rmse = res.fun
        return self.coeffs, self.rmse
