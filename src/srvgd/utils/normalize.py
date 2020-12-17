import numpy as np


def get_normalization_params(data):
    min_ = np.min(data)
    max_ = np.max(data)
    scale_ = 1./(max_-min_)
    return min_, scale_


def normalize(data, min_, scale_):
    return (data-min_)*scale_
