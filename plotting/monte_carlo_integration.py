"""
AUTHOR: Ryan Grindle

LAST MODIFIED: April 7, 2021

PURPOSE: Implement Monte Carlo integration.

NOTES:

TODO:
"""
import numpy as np


def monte_carlo_integration(f, bounds, num_iterations=1000000):
    """
    PARAMETERS
    ----------
    f : function
        The function to integrate
    bounds : np.array
        The bounds of integration as 2D array. Expected shape is (num vars, 2)
    num_iterations : int
        The maximum number of iterations to perform.

    RETURNS
    -------
    integral : float
        The approximate value of the definite integral.
    """
    lengths = bounds[:, 1] - bounds[:, 0]
    input_space_volume = np.product(lengths)

    samples = np.random.uniform(0, 1, size=(num_iterations, len(bounds)))
    samples = lengths*samples + bounds[:, 0]

    return np.mean(f(samples.T))*input_space_volume


def ff_distance(f, g, num_coeffs,
                x_bounds=(0.1, 3.1), c_bounds=(-3, 3)):
    """Expects f, g to take x as first argument
    and then the same number of coefficients. For
    example,
        f = lambda x, c1, c2: c2*x**2 + c1*x
    should be written as
        f = lambda x: x[2]*x[0]**2 + c[1]*x
    """
    num_args_f = f.__code__.co_argcount
    num_args_g = f.__code__.co_argcount
    assert num_args_g == num_args_f

    bounds = np.vstack(((x_bounds,), (c_bounds,)*num_coeffs))
    print(bounds.shape)
    print(bounds)
    diff = lambda x: np.abs(f(x) - g(x))

    return monte_carlo_integration(diff, bounds)


if __name__ == '__main__':
    np.random.seed(0)

    f1 = lambda x:                               x[1]*x[0]
    f2 = lambda x:                x[2]*x[0]**2 + x[1]*x[0]
    f3 = lambda x: x[3]*x[0]**3 + x[2]*x[0]**2 + x[1]*x[0]

    d12 = ff_distance(f1, f2, num_coeffs=3)
    print(d12)

    d13 = ff_distance(f1, f3, num_coeffs=3)
    print(d13)

    d23 = ff_distance(f3, f2, num_coeffs=3)
    print(d23)

    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(1, 2, weight=d12)
    G.add_edge(1, 3, weight=d13)
    G.add_edge(2, 3, weight=d23)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=5000,
            labels={1: '$c_1x$',
                    2: '$c_2x^2 + c_1x$',
                    3: '$c_3x^3 + c_2x^2 + c_1x$'})
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.tight_layout()
    plt.show()
