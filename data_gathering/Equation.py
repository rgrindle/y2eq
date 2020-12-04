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


def dont_recompute_if_exists(func):
    def wrapper(self):
        attr_to_get = func.__name__[4:]
        if hasattr(self, attr_to_get):
            print('why did I need to call this twices')
            print('calling', func.__name__)
            exit()
            return self.__getattribute__(attr_to_get)
        else:
            return func(self)
    return wrapper


class Equation:

    def __init__(self, eq):
        if type(eq) is str:
            eq = sympy.sympify(eq)
        self.eq = eq.expand()
        self.eq = self.get_eq_no_coeff()
        self.eq_f_str = str(self.eq).replace('sin', 'np.sin')
        self.eq_str = str(self.eq).replace('**', '^')
        self.get_func_form()

    def __str__(self):
        return self.eq

    def __repr__(self):
        return 'Equation({})'.format(self)

    def __lt__(self, other):
        return self.eq_str < other.eq_str

    def __eq__(self, other):
        return self.eq_str == other.eq_str

    def get_terms(self):
        return str(self.eq).split(' + ')

    def remove_coeff_mult_at(self, term: str, index: int) -> str:
        end_index = index
        while term[end_index].isdigit():
            end_index += 1
        if end_index == index:
            return term
        else:
            return term[:index] + term[end_index+1:]

    def remove_coeff_term(self, term: str) -> str:
        term = self.remove_coeff_mult_at(term, 0)
        paren_indices = [i+1 for i, t in enumerate(term) if t == '(']
        for i in paren_indices:
            term = self.remove_coeff_mult_at(term, i)
        return term

    @dont_recompute_if_exists
    def get_eq_no_coeff(self) -> str:
        no_coeff_terms = [self.remove_coeff_term(t) for t in self.get_terms()]
        self.eq_no_coeff = '+'.join(no_coeff_terms)
        return self.eq_no_coeff

    @dont_recompute_if_exists
    def get_f(self):
        self.f = eval('lambda x, c: {}'.format(self.func_form))
        return self.f

    def eval(self, X):
        self.get_f()
        return self.f(X, self.coeffs).tolist()

    def get_func_form(self):
        coeff_subscript = 0
        func_form_str = []
        for char in self.eq_f_str:
            if char == 'x':
                func_form_str.append('c[{}]*{}'.format(coeff_subscript, char))
                coeff_subscript += 1
            else:
                func_form_str.append(char)
        self.func_form = ''.join(func_form_str)
        self.num_coeffs = coeff_subscript
        return self.func_form


if __name__ == '__main__':
    x = sympy.symbols('x')
    eq = Equation(2*x**2+x)
    print(eq)
    print(eq.get_func_form())
