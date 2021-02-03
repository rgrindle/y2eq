from equation.Equation import Equation
from check_assert import check_assert

import numpy as np


def test___init__():
    eq = Equation('x**2', apply_coeffs=False)
    yield check_assert, eq.eq_str == 'x**2'
    yield check_assert, eq.f(3) == 9


def test_fig():
    x = np.linspace(-1, 1, 20)
    eq = Equation('c[0]*x**2', x=x, num_coeffs=1,
                  apply_coeffs=False)
    y = 2.5*x**2
    eq.fit(y)
    yield check_assert, eq.rmse <= 10**(-8)
    yield check_assert, eq.coeffs - np.array([3.5]) <= 10**(-8)

    eq = Equation('c[0]*x**2+c[1]', x=x, num_coeffs=2,
                  apply_coeffs=False)
    y = 2.5*x**2-1.234
    eq.fit(y)
    yield check_assert, eq.rmse <= 10**(-8)
    yield check_assert, eq.coeffs - np.array([2.5, 1.234]) <= 10**(-8)
