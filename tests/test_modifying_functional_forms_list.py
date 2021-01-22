from jupyter.modifying_functional_forms_list import fix_ff
from check_assert import check_assert

import pandas as pd


def test_fix_ff():
    corrected_ff = pd.read_csv('jupyter/corrected_ff.csv', header=None).values.flatten()
    not_corrected_ff = pd.read_csv('jupyter/not_corrected_ff.csv', header=None).values.flatten()

    for ff, ans in zip(not_corrected_ff, corrected_ff):
        yield check_assert, fix_ff(ff) == ans
