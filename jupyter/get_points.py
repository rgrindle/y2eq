"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 12, 2021

PURPOSE: Determine how many data points are necessary
         to describe a given function.

NOTES:

TODO:
"""
import numpy as np

import copy


def interpolate(data, new_x):
    assert data.shape == (2, 2)
    x, y = data[0], data[1]
    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[0] - m*x[0]
    return m*new_x + b


def get_points(f, a, b, tolerance=0.1):
    x = get_x(f, a, b, tolerance)
    return x, f(x)


def get_x(f, a, b, tolerance):
    new_x = np.random.uniform(a, b, 3)
    x = [a, b]
    y = f(np.array(x))
    y_pred = interpolate(np.array((x, y)), new_x)
    y_true = f(new_x)
    error = np.sum(np.abs(y_true-y_pred))
    if error <= tolerance:
        return x
    else:
        x = [a] + sorted(new_x) + [b]
        final_x = copy.copy(x)
        for i, _ in enumerate(x[:-1]):
            _x = get_x(f, a=x[i], b=x[i+1], tolerance=tolerance)
            final_x.extend(_x[1:-1])
        return np.unique(final_x)


def is_enough_points(f, x, tolerance):
    for i, _ in enumerate(x[:-1]):
        a, b = x[i], x[i+1]
        new_x = np.random.uniform(a, b, 3)
        sub_x = [a, b]
        y = f(np.array(sub_x))
        y_pred = interpolate(np.array((sub_x, y)), new_x)
        y_true = f(new_x)
        error = np.sum(np.abs(y_true-y_pred))
        if error > tolerance:
            return False
            break
    else:
        return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(1234)

    f = lambda x: np.sin(0.0001*np.exp(3*x)+0.0001*np.exp(2*x)+np.exp(x))
    is_enough = is_enough_points(f, np.arange(0.1, 3.1, 0.1), tolerance=0.1)
    print(is_enough)

    x, y = get_points(f=f, a=0.1, b=3.1, tolerance=0.1)
    print(len(x))

    plt.figure()
    x_many = np.linspace(0.1, 3.1, 10000)
    plt.plot(x_many, f(x_many), label='true')
    plt.plot(x, y, '.-', label='pred')
    plt.legend()
    plt.show()
