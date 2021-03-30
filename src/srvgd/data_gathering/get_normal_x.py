"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Pick x-values from normal distribution and
         ensure that x-values fall in interval.

NOTES:

TODO:
"""
import numpy as np


def reflect_x(x, a=0.1, b=3.1):
    """If any of x fall outside [a, b]
    reflect them back in.
    """
    # reflect on lower bound
    indices_less = a > x
    x[indices_less] = 2*a - x[indices_less]

    # reflect on upper bound
    indices_greater = x >= b
    x[indices_greater] = 2*b - x[indices_greater]

    return x


def get_normal_x(num_points,
                 a=0.1, b=3.1, std_dev_min=0.1,
                 mean_radius=0.5):
    mean = (a+b)/2
    mean = np.random.uniform(mean-mean_radius, mean+mean_radius)
    std_dev_max = min([(mean-a)/3, (b-mean)/3])
    std_dev = np.random.uniform(std_dev_min, std_dev_max)
    x = np.random.normal(loc=mean, scale=std_dev, size=num_points)

    # Now, ensure that all x are in [a, b]
    return reflect_x(x, a=a, b=b)


if __name__ == '__main__':
    x = get_normal_x(num_points=1000, mean_radius=0)
