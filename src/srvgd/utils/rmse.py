"""
AUTHOR: Ryan

LAST MODIFIED: Apr 13, 2021

PURPOSE: Implement root mean squared error.

NOTES:

TODO:
"""

import numpy as np


def RMSE(y, y_hat):
    assert y.shape == y_hat.shape, 'y.shape = '+str(y.shape)+', y_hat.shape = '+str(y_hat.shape)
    return np.sqrt(np.mean(np.power(y-y_hat, 2)))
