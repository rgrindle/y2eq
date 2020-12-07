from data_gathering.Equation import Equation
from check_assert import check_assert


def test_positive_const_eq():
    E = Equation('0')
    yield check_assert, str(E) == '0'
    yield check_assert, E.eq_str == '0'
    yield check_assert, E.eq_f_str == '0*x0[0]'
    yield check_assert, E.func_form == '0*x0[0]+c[0]'
    yield check_assert, E.num_coeffs == 1


def test_coeff_removal():
    E = Equation('2*x0**2')
    yield check_assert, str(E.eq) == 'x0**2'


def test_multi_term_coeff_removal():
    E = Equation('2*x0**2+3*x0+9')
    yield check_assert, str(E.eq) == 'x0**2+x0'


def test_multi_term_coeff_removal_log():
    E = Equation('log(2*x0)')   # == 'log(x) + log(2)
    yield check_assert, str(E.eq) == 'log(x0)'


def test_multi_term_coeff_removal_log2():
    E = Equation('x0*log(2*x0)')
    # after expand, we get x0*log(x0) + x0*log(2)
    yield check_assert, str(E.eq) == 'x0*log(x0)+x0'


def test_functional_form():
    E = Equation('x0**2+x0')
    yield check_assert, E.func_form == 'c[0]*x0**2+c[1]*x0+c[2]'
