"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 18, 2021

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


if __name__ == '__main__':
    np.random.seed(0)

    f = lambda x: x[0]**2+x[1]
    ans = monte_carlo_integration(f,
                                  bounds=np.array([[0, 1],
                                                   [-1, 1]]))
    print(ans)
    # expect ans = 2/3