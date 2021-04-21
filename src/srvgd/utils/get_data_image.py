"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 9, 2021

PURPOSE: Make a image of data points. This will be used as
         input to a NN.

NOTES:

TODO:
"""
import numpy as np

from itertools import product


def get_data_image(data, dim=2, bins=(64, 64)):
    """Convert data points into image.

    Parameters
    ----------
    data : 2D list or array - data.shape = (n, dim)
        Each row is a single observation. First column(s)
        are input(s) and the last column is output.
    dim : int
        The number of dimensions need for the data.
    bin : tuple of length dim
        The number of bins in each dimension.

    Return
    ------
    hist : 3D np.array
        Matrix of ones and zeros indicating there the points
        fall.
    """
    data = np.array(data)
    assert data.shape[1] == dim, 'Each data point should have length dim='+str(dim)
    assert len(bins) == dim, 'There should be one bin size for each dim='+str(dim)
    for d in range(dim-1):
        assert 0.1 <= np.min(data[:, d]) and np.max(data[:, d]) <= 3.1, 'input must be in [0.1, 3.1]: '+str(data[:, d])
    assert 0.0 <= np.min(data[:, -1]) and np.max(data[:, -1]) <= 1.0, 'output must be in [0, 1]: '+str(data[:, -1])

    hist, _ = np.histogramdd(data, bins=bins,
                             range=[[0.1, 3.1]]*(dim-1) + [[0.0, 1.0]])
    hist = np.where(hist >= 1, 1, 0)

    if dim == 2:
        # Make 3 channel since
        # that is what ResNet expects.
        # Just make each channel the same.
        hist = hist.T[None]
        hist = np.repeat(hist, 3, axis=0)
    return hist


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
    from plot2eq.utils import normalize

    x = get_2d_grid(1024).T
    print(x.shape)
    y = normalize(np.sin(x[0])+np.sin(3*x[1]))
    print(y.shape)
    points = np.vstack((x, y[None, :])).T
    print(points.shape)

    H = get_data_image(points, dim=3, bins=(35, 35, 64))

    # H, edges = np.histogramdd(points, bins=(35, 35, 64))

    print(H)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(H, facecolors='C0', edgecolor='C0')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    plt.show()
