"""
AUTHOR: Ryan Grindle

PURPOSE: Store an equation in many forms for use by
         DatasetGenerator. Also, handle some equation
         manipulation like adding coefficients for
         example.

NOTES: It is assumed that eq is never changed.

TODO:
"""
import sympy  # type: ignore
import numpy as np  # noqa: F401 # type: ignore

from typing import List, Tuple


def dont_recompute_if_exists(func):
    def wrapper(self):
        attr_to_get = func.__name__[4:]
        if hasattr(self, attr_to_get):
            return self.__getattribute__(attr_to_get)
        else:
            return func(self)
    return wrapper


x0 = sympy.symbols('x0', real=True)


class Equation:

    def __init__(self, eq):
        if eq == '0':
            self.eq = '0'
            self.eq_str = '0'
            self.eq_f_str = '0*x0[0]'
            self.func_form = '0*x0[0]+c[0]'
            self.num_coeffs = 1
        else:
            if type(eq) is str:
                for prim in ['sin', 'exp', 'log', 'cos(', 'cosh']:
                    eq = eq.replace(prim, 'sympy.'+prim)
                eq = eval(eq)
            self.eq = eq.expand()
            self.eq = self.get_eq_no_coeff()
            self.eq_f_str = str(self.eq)
            for prim in ['sin', 'exp', 'log', 'cos(', 'cosh']:
                self.eq_f_str = self.eq_f_str.replace(prim, 'np.'+prim)
            self.eq_str = str(self.eq).replace('**', '^')
            self.get_func_form()

    def __str__(self):
        return self.eq_str

    def __repr__(self):
        return 'Equation({})'.format(self)

    def __lt__(self, other):
        return self.eq_str < other.eq_str

    def __eq__(self, other):
        return self.eq_str == other.eq_str

    def get_terms(self, expr=None):
        expr = self.eq if expr is None else expr
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
        end_index = index
        while end_index < len(term) and term[end_index].isdigit():
            end_index += 1
        if end_index == index:
            return term
        else:
            return term[:index] + term[end_index+1:]

    def remove_coeff_term(self, term: str) -> str:
        if 'x0' not in term:
            return ''

        if '+' not in term:
            term = self.remove_coeff_mult_at(term, 0)
            matched_paren_ind = self.get_matched_paren_ind(term)
            # remove hidden coeffs (e.g. x*log(2) -> x)
            for lf_i, rt_i in matched_paren_ind:
                if 'x0' not in term[lf_i+1:rt_i]:
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
            try:
                # this removes vertical shifts
                snc.remove('')
            except ValueError:
                pass
            return term[:lf_i+1] + '+'.join(snc) + term[rt_i:]

    @dont_recompute_if_exists
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

    @dont_recompute_if_exists
    def get_f(self):
        try:
            self.f = eval('lambda x0, c: {}'.format(self.func_form))
            return self.f
        except SyntaxError as e:
            print(str(e))
            print(self.eq)
            exit()

    def eval(self, X):
        self.get_f()
        return self.f(X, self.coeffs).tolist()

    def get_func_form(self):
        coeff_subscript = 0
        func_form_list = []
        for i, char in enumerate(self.eq_f_str):
            if char == 'x' and self.eq_f_str[i+1] == '0':
                func_form_list.append('c[{}]*{}'.format(coeff_subscript, char))
                coeff_subscript += 1
            else:
                func_form_list.append(char)
        func_form_list.append('+c[{}]'.format(coeff_subscript))
        self.func_form = ''.join(func_form_list)
        self.num_coeffs = coeff_subscript+1
        return self.func_form


if __name__ == '__main__':
    x = sympy.symbols('x0', real=True)
    eq = Equation('exp(x0 + sin(x0))')
    print(eq)
    print(eq.get_func_form())
    # exp(2*exp(x0)) exp(e
