"""
AUTHOR: Ryan Grindle

PURPOSE: Understand SymPy

NOTES: Quote from https://docs.sympy.org/latest/tutorial/manipulation.html
       "The arguments of the commutative operations Add and Mul are stored
       in an arbitrary (but consistent!) order, which is independent of the
       order inputted"

TODO:
"""


import sympy


def get_func_form(expr):
    assert len(expr.args) <= 10
    return '+'.join(['c{}*{}'.format(i, a) for i, a in enumerate(expr.args)])


if __name__ == '__main__':
    x = sympy.symbols('x')
    expr = x*sympy.exp(x)+x**2+x+x**2+x**3 + sympy.sin(x*x)
    print(expr)
    print(expr.args)
    func_form = get_func_form(expr)
    # print(sympy.srepr(expr))
    print(func_form)
