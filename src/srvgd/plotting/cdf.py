"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 14, 2021

PURPOSE: Provide functions to plot cumulative distribution.

NOTES:

TODO:
"""

import numpy as np
import matplotlib.pyplot as plt


def get_cdf(X):
    """Get the probability that x_i > X after X has
    been sorted (that is x_i is >= i+1 other x's).
    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF
    Returns
    -------
    p : list
        A list of the same length as X that
        give the probability that x_i > X
        where X is a randomly selected value
        and i is the index.
    """

    X_sorted = sorted(X)
    n = len(X)
    p = [i/n for i, x in enumerate(X)]
    return p, X_sorted


def plot_cdf(X, labels=True, **kwargs):
    """Use get_cdf to plot the CDF.
    Parameters
    ----------
    X : list
        A sample from the distribution for
        which to compute the CDF
    labels : bool (default=True)
        If true, label x-axis x and y-axis Pr(X < x)
    label : str (default=None)
        The legend label.
    color : str (default=None)
        Color to used in plot. If none, it will not
        be pasted to plt.step.
    """

    p, X = get_cdf(X)

    plt.fill_between(X, len(X)*np.array(p),
                     **kwargs)

    if labels:
        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')
