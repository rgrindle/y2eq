import numpy as np  # type: ignore


def check_assert(condition, message=''):
    assert np.all(condition), message
