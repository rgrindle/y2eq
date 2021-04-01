from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq, get_eq_string
from check_assert import check_assert


def test_tokenzie_eq():
    yield check_assert, tokenize_eq('x') == [12, 1, 13]
    yield check_assert, tokenize_eq('x0**x1', two_d=True) == [12, 16, 7, 17, 13]


def test_get_eq_string():
    yield check_assert, get_eq_string([12, 1, 13]) == 'STARTxEND'
    yield check_assert, get_eq_string([12, 16, 7, 17, 13], two_d=True) == 'STARTx0**x1END'
