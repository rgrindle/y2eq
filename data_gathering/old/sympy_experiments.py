"""
AUTHOR: Ryan Grindle

PURPOSE: Understand SymPy

NOTES: Quote from https://docs.sympy.org/latest/tutorial/manipulation.html
       "The arguments of the commutative operations Add and Mul are stored
       in an arbitrary (but consistent!) order, which is independent of the
       order inputted"

TODO:
"""


import sympy  # type: ignore


def get_func_form(expr):
    assert len(expr.args) <= 10
    return '+'.join(['c{}*{}'.format(i, a) for i, a in enumerate(expr.args)])


def remove_coeff_term(term):
    term_str = str(term)

    if term_str[0].isdigit():
        return term_str[2:]
    else:
        return term_str


def remove_coeff(expr):
    no_coeff_terms = [remove_coeff_term(t) for t in str(expr).split('+')]
    return sympy.sympify('+'.join(no_coeff_terms))


if __name__ == '__main__':
    x = sympy.symbols('x')

    expr = x+x + 2*x**2
    # expr = 2*x
    print('expr', expr)

    no_coeff_expr = remove_coeff(expr)
    print('no_coeff_expr', no_coeff_expr)

    import numpy as np  # type: ignore
    lambdafied = np.vectorize(sympy.utilities.lambdify(x, no_coeff_expr))
    print('lambdafied', lambdafied)
    print('lambdafied(3)', lambdafied(3))
    print('lambdafied([1, 2, 3])', lambdafied([1, 2, 3]))

    # expr = x*sympy.exp(x)+x**2+x+x**2+x**3 + sympy.sin(x*x)
    # print(expr)
    # print(expr.args)
    # func_form = get_func_form(expr)
    # print(sympy.srepr(expr))
    # print(func_form)
