"""
AHUTOR: Ryan Grindle

LAST MODIFIED: April 1, 2021

PURPOSE: Normalize data into [0, 1] and optionally
         get the parameters the perform this normalization.
         Can also used unnormalize (and the correct params)
         to undo a normalization.

NOTES:

TODO:
"""

import numpy as np


def get_normalization_params(data):
    min_ = np.min(data)
    max_ = np.max(data)
    scale_ = 1./(max_-min_)
    return min_, scale_


def normalize(data, min_=None, scale_=None, return_params=False):
    if min_ is None:
        assert scale_ is None, 'Either both min_, scale_ are None or neither'
        min_, scale_ = get_normalization_params(data)

    normalized_data = (data-min_)*scale_

    if return_params:
        return normalized_data, min_, scale_
    else:
        return normalized_data


def unnormalize(data, min_data, scale_):
    return data/scale_+min_data
