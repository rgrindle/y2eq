import numpy as np


def get_normalization_params(data):
    min_ = np.min(data)
    max_ = np.max(data)
    scale_ = 1./(max_-min_)
    return min_, scale_


def normalize(data, min_=None, scale_=None):
    if min_ is None:
        assert scale_ is None, 'Either both min_, scale_ are None or neither'
        min_, scale_ = get_normalization_params(data)
    return (data-min_)*scale_
