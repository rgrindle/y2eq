import matplotlib.pyplot as plt


def get_cumulative_distribution_funtion(X):
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


def plot_cumulative_distribution_funtion(X,
                                         labels=True,
                                         label=None,
                                         color=None):
    """Use get_emprical_cumulative_distribution_funtion to plot the CDF.
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

    p, X = get_cumulative_distribution_funtion(X)

    if color is None:
        plt.fill_between(X, len(X)*np.array(p),
                         step='post', label=label)
    else:
        plt.fill_between(X, len(X)*np.array(p),
                         step='post', label=label,
                         color=color)

    if labels:
        plt.ylabel('$Pr(X < x)$')
        plt.xlabel('$x$')


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    rmse_data = pd.read_csv('rmse_test.csv').values[:, 1]
    mask = ~(np.logical_or(np.isnan(rmse_data), np.isinf(rmse_data)))
    print(rmse_data[mask])
    plot_cumulative_distribution_funtion(rmse_data[mask], labels=False, color='#8B94FC')
    plt.xlabel('RMSE')
    plt.ylabel('Cumulative Counts')
    # plt.xlim([0, 3])
    # plt.ylim([0, 1000])
    plt.savefig('figure_recreation2.pdf')
