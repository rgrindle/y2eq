"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 14, 2021

PURPOSE: Take existing dataset where input is
         30 y-values for each observation
            (shape = (num obs, 30, 1))
         and change it to images.

NOTES:

TODO:
"""
from srvgd.utils.get_data_image import get_data_image

import torch
from torch.utils.data import TensorDataset
import numpy as np


def change_input_to_image(dataset, image_size):

    # get y and functional forms from dataset
    y = [d[0].tolist() for d in dataset]
    y = torch.Tensor(y)
    ff = [d[1].tolist() for d in dataset]

    # make y into images
    x = torch.arange(0.1, 3.1, 0.1)[None]
    x = torch.repeat_interleave(x, len(y), dim=0)
    points = torch.stack((x, y[:, :, 0]), axis=-1)
    imgs = [np.flip(get_data_image(p, bins=image_size), axis=1) for p in points]

    # turn into dataset
    return TensorDataset(torch.Tensor(imgs), torch.LongTensor(ff))
