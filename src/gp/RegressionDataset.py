"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Oct 16, 2020

PURPOSE: This file contains a class to deal with regression
         datasets. It holds x, y data, which can be generated
         via function f if desired. There are also functions
         to generated x in the correct shape, check the
         x and y are compatible, and get errors.

NOTES:

TODO:
"""
from gp.protected_functions import protected_exp, protected_log  # noqa: F401

import numpy as np  # type: ignore
import matplotlib.pyplot as plt

import itertools


class RegressionDataset:
    """Container for regression dataset"""

    def __init__(self, x, y=None, f=None):
        """For some unknown function f these quantities
        are related by y = f(x) + e where e is noise.

        Parameters
        ----------
        x : np.array
            2D array where each row is an observation
            and each column is variable (x0, x1, ...)
        y : np.array
            2D array where each row is an observation
            and there is only one column.
        f : function
            Maps x to y. That is, y = f(x)
        """
        self.x = np.array(x)

        if y is not None:
            self.y = np.array(y)
            self.is_valid(self.x, self.y)

        if f is not None:
            self.f = f
            self.y = self.get_y(self.x, self.f)
            self.is_valid(self.x, self.y)

    @staticmethod
    def is_valid(x, y):
        assert len(x.shape) == 2, 'Expected x to be 2D.'
        assert len(y.shape) == 2, 'Expected y to be 2D.'
        assert x.shape[0] == y.shape[0], 'Expected same number of observations (rows) in x and y.'

    @staticmethod
    def linspace(a: int, b: int, n: int, num_vars: int = 1):
        """
        Parameters
        ----------
        a : float
            The left end point.
        b : float
            The right end point.
        n : int
            The total number of points.
        num_vars : int
            The number of input variables.

        Returns
        -------
        x : np.array
            2D array where the number of columns is num_vars
            and the number of rows is n.
        """
        assert n//num_vars == n/num_vars, 'num_vars does not divide n'
        x_vars = [np.linspace(a, b, n//num_vars) for _ in range(num_vars)]
        return np.array(list(itertools.product(*x_vars)))

    @staticmethod
    def urandspace(a: int, b: int, n: int, num_vars: int = 1):
        """
        Parameters
        ----------
        a : float
            The left end point.
        b : float
            The right end point.
        n : int
            The total number of points.
        num_vars : int
            The number of input variables.

        Returns
        -------
        x : np.array
            2D array where the number of columns is num_vars
            and the number of rows is n.
        """
        assert n//num_vars == n/num_vars, 'num_vars does not divide n'
        x_vars = [np.random.uniform(a, b, n//num_vars) for _ in range(num_vars)]
        return np.array(list(itertools.product(*x_vars)))

    @staticmethod
    def get_y(x, f, c=None):
        """Get y = f(x)

        Parameters
        ----------
        x : np.array
            A 2D array where columns are variables (x[0], x[1], ...)
            and rows are observations
        f : function
            The function that maps each row of x to a value of y.
        Returns
        -------
        y : np.array
            A 2D array where there is only one column and each row
            corresponds to the same row in x.
        """
        assert len(x.shape) == 2, 'x must be 2D'

        if c is None:
            y = f(x.T)
        else:
            y = f(c=c, x=x.T)
        return y[:, None]  # make 2D

    def get_dataset(self):
        return np.hstack((self.x, self.y))

    def get_RMSE(self, f_hat=None, y_hat=None):
        assert (f_hat is not None) != (y_hat is not None), 'set either f_hat or y_hat, but not both'
        if f_hat is not None:
            y_hat = RegressionDataset.get_y(self.x, f_hat)
        return self.get_RMSE_static(self.y, y_hat)

    @staticmethod
    def get_RMSE_static(y, y_hat):
        assert np.all(y.shape == y_hat.shape), 'y {} and y_hat {} must have the same shape'.format(y.shape, y_hat.shape)
        return np.sqrt(np.mean(np.power(y_hat-y, 2)))

    def get_NRMSE(self, f):
        y_hat = RegressionDataset.get_y(self.x, f)
        return RegressionDataset.get_NRMSE_static(y=self.y, y_hat=y_hat)

    @staticmethod
    def get_NRMSE_static(y, y_hat):
        RMSE = RegressionDataset.get_RMSE_static(y, y_hat)
        denom = np.max(y)-np.min(y)

        # if constant function do normal RMSE
        if denom == 0:
            denom = 1

        return RMSE/denom

    def zscore_inverse(self, z):
        if self.std != 0:
            return z*self.std + self.mean
        else:
            return z

    def get_signed_error(self, f_hat):
        y_hat = self.get_y(self.x, f_hat)
        return (self.y - y_hat).flatten()

    def plot(self):
        assert self.x.shape[1] == 1, 'Cannot make 2D plot'
        plt.plot(self.x, self.y, '.-')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        # plt.show()

    def __eq__(self, other):

        if np.any(self.x != other.x):
            return False
        elif np.any(np.abs(self.y - other.y) >= 10**(-10)):
            return False
        else:
            return True

    def __len__(self):
        return self.y.shape[0]
