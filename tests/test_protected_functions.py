import gp.protected_functions as pf
from check_assert import check_assert

import numpy as np


def test_protected_log():
    ans = pf.protected_log([-3, 0, 1e-1000000])
    yield check_assert, np.logical_not(np.isnan(ans))
    yield check_assert, np.logical_not(np.isinf(ans))


def test_protected_exp():
    ans = pf.protected_exp([-100000, -101, -100, -99, -1, 0, 1, 99, 100, 101, 100000])
    yield check_assert, np.logical_not(np.isnan(ans))
    yield check_assert, np.logical_not(np.isinf(ans))


def test_protected_pow():
    ans = pf.protected_pow([-100000, -101, -100, -99, -1, 0, 1, 99, 100, 101, 100000], 6)
    yield check_assert, np.logical_not(np.isnan(ans))
    yield check_assert, np.logical_not(np.isinf(ans))


def test_protected_div():
    ans = pf.protected_div([-10, 0, 5, 10], [-1e-10000, 0, 0, 1e-10000])
    yield check_assert, np.logical_not(np.isnan(ans))
    yield check_assert, np.logical_not(np.isinf(ans))
