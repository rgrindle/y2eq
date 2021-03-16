"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 2, 2021

PURPOSE: Given a number of points, make a uniformly
         spaced grid of points in the interval
         [0.1, 3.1) x [0.1, 3.1).

NOTES:

TODO:
"""
import numpy as np

from itertools import product


def get_2d_grid(num_points, a=0.1, b=3.1):
    """To make a square grid,
    num_points must be a perfect
    square. For now, this is required.
    """
    # Note: points_per_row = points_per_col
    points_per_row = np.sqrt(num_points)
    assert points_per_row == int(points_per_row)
    x0 = np.arange(a, b, (b-a)/points_per_row)
    x1 = np.arange(a, b, (b-a)/points_per_row)
    return np.array(list(product(x0, x1)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    grid = get_2d_grid(1024)
    print(grid.shape)

    f = lambda x: x[0]*np.sin(x[1])

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(grid[:, 0], grid[:, 1], f(grid.T))
    plt.show()
