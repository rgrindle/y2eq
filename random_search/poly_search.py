"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 24, 2021

PURPOSE: Generate random polynomials (functional forms).
         Will use BFGS to fit coefficients. Want to know
         if this method can compete with y2eq.

NOTES:

TODO:
"""
import numpy as np

np.random.seed(0)


def get_poly(num):
    """Convert a number to a polynomial
    using binary to indicate the presence
    of each term."""
    assert 1 <= num <= 128
    binary_num = '{0:07b}'.format(num)
    print(binary_num)
    poly_list = ['x**{}'.format(i) for i in range(7) if binary_num[6-i] == '1']
    print(poly_list)
    poly_str = ' + '.join(poly_list)
    print(poly_str)
    return poly_str


def get_rand_poly():
    """Gererate polynomial with degree of
    0 at smallest and 6 at largest. All
    coefficients will be 1."""
    poly_num = np.random.randint(1, 129)
    print(poly_num)
    return get_poly(poly_num)


if __name__ == '__main__':
    get_rand_poly()
