"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 2, 2021

PURPOSE: This class takes in a functional form and applies
         coefficients. Then, it can use y-values to determine
         the best coefficient values.

NOTES:

TODO:
"""
from equation.Equation import Equation
import re


class EquationInfix(Equation):

    def apply_coeffs(self):
        """Take eq (str without coeffs although possibly
        hard-coded ones) and put c[0], c[1], ... where
        applicable.

        Returns
        -------
        eq_c : str
            equation str including adjustable coefficients.
        num_coeff : int
            The number of coefficients placed.

        Examples
        --------
        >>> apply_coeffs('x')
        ('c[0]*x', 1)

        >>> apply_coeffs('sin(x)')
        ('c[1]*sin(c[0]*x)', 2)

        >>> apply_coeffs('sin(exp(x))')
        ('c[1]*sin(c[2]*exp(c[0]*x))', 3)
        """
        self.eq_ff = self.eq_str
        coeff_index = 0

        # First, attach a coefficient to every occurance of x.
        # Be careful to find the variable not the x in exp, for example.
        eq_str_list = []
        for i, e in enumerate(self.eq_str):
            if e == 'x' and (i == 0 or self.eq_str[i-1] != 'e'):
                eq_str_list.append('c[{}]*x'.format(coeff_index))
                coeff_index += 1
            else:
                eq_str_list.append(e)

        # Put a coefficient in front of every term.
        c_eq = ''.join(eq_str_list)
        c_eq_str_list = []
        for term in c_eq.split('+'):
            if 'c[' == term[0:2]:
                c_eq_str_list.append(term)
            else:
                c_eq_str_list.append('c[{}]*'.format(coeff_index)+term)
                coeff_index += 1
        c_eq = '+'.join(c_eq_str_list)

        # Put a coeff in front of any missed primitives.
        # Without this block sin(sin(x)) -> c[1]*sin(sin(c[0]*x))
        # but with this block sin(sin(x)) -> c[1]*sin(c[2]*sin(c[0]*x))
        for prim in ['sin', 'exp', 'log']:
            c_eq_str_list = []
            prev_i = 0
            for m in re.finditer(prim, c_eq):
                i = m.start()
                c_eq_str_list.append(c_eq[prev_i:i])
                if c_eq[i-2:i] != ']*':
                    c_eq_str_list.append('c[{}]*'.format(coeff_index))
                    coeff_index += 1
                prev_i = i
            c_eq_str_list.append(c_eq[prev_i:])
            c_eq = ''.join(c_eq_str_list)

        self.eq_str = c_eq
        self.num_coeffs = coeff_index
