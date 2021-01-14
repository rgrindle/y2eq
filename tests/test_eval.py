import srvgd.utils.eval as e
from check_assert import check_assert


def test_apply_coeffs():
    eq_with_coeff, num_coeff = e.apply_coeffs('x')
    yield check_assert, eq_with_coeff == 'c[0]*x'
    yield check_assert, num_coeff == 1

    eq_with_coeff, num_coeff = e.apply_coeffs('x+x**2')
    yield check_assert, eq_with_coeff == 'c[0]*x+c[1]*x**2'
    yield check_assert, num_coeff == 2

    eq_with_coeff, num_coeff = e.apply_coeffs('sin(x)')
    yield check_assert, eq_with_coeff == 'c[1]*sin(c[0]*x)'
    yield check_assert, num_coeff == 2

    eq_with_coeff, num_coeff = e.apply_coeffs('sin(exp(x))')
    yield check_assert, eq_with_coeff == 'c[1]*sin(c[2]*exp(c[0]*x))'
    yield check_assert, num_coeff == 3
