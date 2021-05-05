"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Figure out in what way invalid equations output by
         the NN are invalid. If possible, fix them.

NOTES:

TODO:
"""
from equation.EquationInfix import EquationInfix

to_try = [('add', 1, ')'),
          ('remove', 1)]

counts = {t: 0 for t in to_try}
counts['valid'] = 0


def attempt_to_make_valid(eq_str, x_int):
    eq = EquationInfix(eq_str, x=x_int, apply_coeffs=False)

    if eq.is_valid():
        return eq

    global to_try, counts

    for t in to_try:
        cmd = t[0]
        num = t[1]
        if cmd == 'add':
            token = t[2]
            new_eq_str = eq_str + token*num
        elif t[0] == 'remove':
            new_eq_str = eq_str[:-num]
        else:
            print('ERROR: unknown rule in position 0 of tuple: '+cmd)

        eq = EquationInfix(new_eq_str, x=x_int, apply_coeffs=False)

        if eq.is_valid():
            counts[t] += 1
            counts['valid'] += 1
            return eq
