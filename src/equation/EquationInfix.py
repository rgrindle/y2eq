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

from typing import Tuple, List


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

    def get_terms(self, expr=None):
        expr = self.eq_str if expr is None else expr
        eq_str = str(expr).replace(' ', '')
        split_ind = [-1]
        paren_ct = {'left': 0, 'right': 0}
        for i, s in enumerate(eq_str):
            if s == '(':
                paren_ct['left'] += 1
            elif s == ')':
                paren_ct['right'] += 1
            if s == '+' and paren_ct['left'] == paren_ct['right']:
                split_ind.append(i)
        split_ind.append(len(eq_str))
        return [eq_str[s+1:e] for s, e in zip(split_ind, split_ind[1:])]

    def remove_coeff_mult_at(self, term: str, index: int) -> str:
        _, end_index = self.get_number(term, index)
        if end_index == index:
            return term
        else:
            if end_index+1 > len(term):
                subterm = term[:index-1] + term[end_index:]
            elif index == 0:
                subterm = term[:index] + term[end_index+1:]
            else:
                if term[index-1] == '*':
                    subterm = term[:index-1] + term[end_index:]
                elif term[end_index] == '*':
                    subterm = term[:index] + term[end_index+1:]
                else:
                    print('ERROR: Cannot find * in term')
                    exit()
            return self.remove_coeff_mult_at(subterm, index)

    def get_number(self, string, start_index=0):
        end_index = start_index
        while end_index < len(string) and (string[end_index].isdigit() or string[end_index] in ('.', '-')):
            end_index += 1
        return start_index, end_index

    def handle_constant_terms(self, term):
        if 'x[0]' not in term and 'x[1]' not in term:
            try:
                mult_index = term.index('*')

                # check if after * is 1
                a_s_index, a_e_index = self.get_number(term, mult_index+1)
                if term[a_s_index:a_e_index] == '1':
                    return '1'

                # check if before * is 1
                b_s_index, b_e_index = self.get_number(term[:mult_index][::-1])

                b_s_index, b_e_index = mult_index-b_e_index, mult_index
                if term[b_s_index:b_e_index] == '1':
                    return '1'
                else:
                    return ''

            except ValueError:
                return '1' if term == '1' else ''

    def remove_coeff_term(self, term: str) -> str:
        const = self.handle_constant_terms(term)
        if const is not None:
            return const

        term = self.remove_coeff_mult_at(term, 0)

        # remove multiplied coefficients (e.g. x[1]*2*x[0] -> x[1]*x[0])
        mult_locs = [m.start(0) for m in re.finditer('(?<!\\*)\\*(?!\\*)', term)]
        for i in reversed(mult_locs):
            if term[i] == '*':
                if self.handle_constant_terms(term[i-1:]) != '1':
                    term = self.remove_coeff_mult_at(term, i+1)

        if '+' not in term:
            matched_paren_ind = self.get_matched_paren_ind(term)
            # remove hidden coeffs (e.g. x*log(2) -> x)
            for lf_i, rt_i in matched_paren_ind:
                if 'x[0]' not in term[lf_i+1:rt_i] and 'x[1]' not in term[lf_i+1:rt_i]:
                    if term[lf_i-4] == '*':
                        term = term[:lf_i-4] + term[rt_i+1:]
                    else:
                        term = term[:lf_i-3] + term[rt_i+2:]
                    break

            # remove more obvious coeffs (e.g. log(2*x) -> log(x))
            for lf_i, _ in matched_paren_ind:
                term = self.remove_coeff_mult_at(term, lf_i+1)

            return term
        else:
            matched_paren_ind = self.get_matched_paren_ind(term)
            for lf_i, rt_i in matched_paren_ind:
                subterm_list = self.get_terms(term[lf_i+1:rt_i])
            snc = [self.remove_coeff_term(subterm) for subterm in subterm_list]
            return term[:lf_i+1] + '+'.join(snc) + term[rt_i:]

    def get_eq_no_coeff(self) -> str:
        no_coeff_terms = [self.remove_coeff_term(t) for t in self.get_terms()]
        try:
            # this removes vertical shifts
            no_coeff_terms.remove('')
        except ValueError:
            pass
        self.eq_no_coeff = '+'.join(no_coeff_terms)
        return self.eq_no_coeff

    def get_matched_paren_ind(self, term: str) -> List[Tuple[int]]:
        paren_ct = {'left': 0, 'right': 0}
        left_paren_ind = []
        matched_paren_ind = []
        for i, s in enumerate(term):
            if s == '(':
                paren_ct['left'] += 1
                left_paren_ind.append(i)
            elif s == ')':
                paren_ct['right'] += 1
                matched_paren_ind.append((left_paren_ind.pop(), i))
        return matched_paren_ind


if __name__ == '__main__':
    # test 0
    eq = EquationInfix('x[0]+1', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[0]+1'

    eq = EquationInfix('x[0]+0.124*1', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[0]+1'

    eq = EquationInfix('x[0]+1*0.124', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[0]+1'

    eq = EquationInfix('1.234*1', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == '1'

    eq = EquationInfix('1*1.234', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == '1'

    # test 1
    eq = EquationInfix('x[1]*2*x[0]**2', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[1]*x[0]**2'

    # test 1
    eq = EquationInfix('3*x[1]*2*x[0]*8', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[1]*x[0]'

    # test 2
    eq = EquationInfix('3*(4*(2*x[0]+5*x[0]))', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == '((x[0]+x[0]))'

    # test 3
    eq = EquationInfix('exp(3*-1.753*x[1])', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'np.exp(x[1])'

    # test 4
    eq = EquationInfix('-0.381*sin(1.990*log(0.553*x[1])**3)', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'np.sin(np.log(x[1])**3)'

    # test 5
    eq = EquationInfix('(1.23*x[0])', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == '(x[0])'

    eq = EquationInfix('(x[0]*1.23)', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == '(x[0])'

    eq = EquationInfix('x[1]*-1.598*exp(log(x[0])**3+-0.917*1)', apply_coeffs=False)
    assert eq.get_eq_no_coeff() == 'x[1]*np.exp(np.log(x[0])**3+1)'
