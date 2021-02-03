"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 1, 2021

PURPOSE: This class takes in a functional form and applies
         coefficients. Then, it can use y-values to determine
         the best coefficient values.

NOTES:

TODO:
"""
from equation.Equation import Equation
import re


class EquationLisp(Equation):

    def apply_coeffs(self):
        self.eq_ff = self.eq_str
        coeff_index = 0

        # put coeff in front of every x
        eq_str_list = []
        for i, e in enumerate(self.eq_str):
            if e == 'x' and (i == 0 or self.eq_str[i-1] != 'e'):
                eq_str_list.append('c[{}]*x'.format(coeff_index))
                coeff_index += 1
            else:
                eq_str_list.append(e)
        c_eq = ''.join(eq_str_list)

        # put coeff in front of every primitive
        for prim in ['sin', 'protected_exp', 'protected_log']:
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
            self.eq_str = ''.join(c_eq_str_list)
            self.num_coeffs = coeff_index
