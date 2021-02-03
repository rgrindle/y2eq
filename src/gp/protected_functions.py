"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: Define protected functions for use by GP.

NOTES: A protected function is one that cannot return nan
       for any input.

TODO:
"""
import numpy as np


def protected_log(x):
    with np.errstate(invalid='ignore', divide='ignore'):
        ans = np.log(np.abs(x))
        ans[np.isnan(ans)] = 0.
        ans[np.isinf(ans)] = 0.
    return ans


def protected_exp(x):
    """protected_exp(x) = exp(x) if x < 100 else 0"""
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), np.exp(100.))


def protected_div(n, d):
    with np.errstate(divide='ignore', invalid='ignore'):
        quotient = np.divide(n, d)
        quotient[np.isnan(quotient)] = 0.
        quotient[np.isinf(quotient)] = 0.
    return quotient


def protected_pow(base, exponent):
    """Check base is not to big. Assuming
    that exponent in controlled."""
    with np.errstate(over='ignore'):
        return np.where(np.abs(base) < 100, np.power(base, exponent), np.power(100, exponent))


def pow2(base):
    return protected_pow(base, exponent=2)


def pow3(base):
    return protected_pow(base, exponent=3)


def pow4(base):
    return protected_pow(base, exponent=4)


def pow5(base):
    return protected_pow(base, exponent=5)


def pow6(base):
    return protected_pow(base, exponent=6)
