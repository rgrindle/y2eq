from data_gathering.Equation import Equation
from check_assert import check_assert


def test_positive_const_eq():
    E = Equation('0')
    yield check_assert, str(E) == '0'
    yield check_assert, E.eq_str == '0'
    yield check_assert, E.eq_f_str == '0*x[0]'
    yield check_assert, E.func_form == '0*x[0]+c[0]'


def test_coeff_removal():
    E = Equation('2*x0**2')
    yield check_assert, str(E.eq) == 'x0**2'


def test_multi_term_coeff_removal():
    E = Equation('2*x0**2+3*x0+9')
    yield check_assert, str(E.eq) == 'x0**2+x0'


def test_functional_form():
    E = Equation('x0**2+x0')
    yield check_assert, E.func_form == 'c[0]*x0**2+c[1]*x0+c[2]'
