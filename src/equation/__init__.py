"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: This class takes in a functional form and applies
         coefficients. Then, it can use y-values to determine
         the best coefficient values.

NOTES:

TODO:
"""


class Equation:

    def __init__(self, eq_str: str, use_coeffs: bool = True):
        self.eq_str = eq_str

        if use_coeffs:
            self.apply_coeffs()

    def apply_coeffs(self):
        pass

    def fit(self, y):
        pass
