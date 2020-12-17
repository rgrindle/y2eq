import src.srvgd.utils.normalize as n
from check_assert import check_assert

import numpy as np


def test_normalize():
    data = [[1, 2, 3],
            [4, 5, 6]]
    min_, scale_ = n.get_normalization_params(data)
    normalized_data = n.normalize(data, min_, scale_)
    yield check_assert, np.all(0 <= normalized_data)
    yield check_assert, np.all(normalized_data <= 1)


def test_get_normalization_params():
    data = [[1, 2, 3],
            [4, 5, 6]]
    min_, scale_ = n.get_normalization_params(data)
    yield check_assert, min_ == 1
    yield check_assert, scale_ == 1/5
