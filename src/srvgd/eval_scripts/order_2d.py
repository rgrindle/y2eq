"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Order 2d points into 1d sequence. Order is essentially
         left to right then bottom to top.

NOTES:

TODO:
"""
import numpy as np


def order_2d(points, bins=35, a=0.1, b=3.1):
    points = np.array(points)
    assert points.shape[1] == 3, 'Must be 3D points (includes output)'
    assert np.all(a <= points[:, :2]) & np.all(points[:, :2] < b), 'points must be in [a, b)'

    bin_bounds = np.linspace(a, b, bins+1)

    seq = []
    for i in range(bins):
        mask = (bin_bounds[i] <= points[:, 1]) & (points[:, 1] < bin_bounds[i+1])
        points_in_bin = points[mask]
        points_in_bin = points_in_bin[points_in_bin[:, 0].argsort()]  # sort in x0
        seq.extend(points_in_bin[:, -1].tolist())

    # import matplotlib.pyplot as plt
    # points = np.array(points)
    # plt.plot(points[:, 0], points[:, 1], '.', label='points')
    # seq = np.array(seq)
    # plt.plot(seq[:, 0], seq[:, 1])
    # plt.xlabel('x0')
    # plt.ylabel('x1')
    # plt.hlines(bin_bounds, *plt.xlim(), linestyle='--', label='bin bounds')
    # plt.legend()
    # plt.show()

    return seq


if __name__ == '__main__':
    points = [(0, 0, 1),
              (1, 0, 3),
              (2, 0, 4),
              (0, 1, 5),
              (1, 1, 2),
              (2, 1, 6),
              (0, 2, 7),
              (1, 2, 8),
              (2, 2, 9)]
    seq = order_2d(points, bins=3, a=0, b=2.5)
    print(seq)

    points = [(0, 0.3, 1),
              (1.1, 0, 3),
              (2, 0.7, 4),
              (0, 1.5, 5),
              (1, 0.3, 2),
              (2, 1, 6),
              (0, 2, 7),
              (1, 2.3, 8),
              (2, 2.4, 9)]
    seq = order_2d(points, bins=3, a=0, b=2.5)
    print(seq)
