"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Create a version of mse loss function
         that will ignore certain elements. For example,
         if the desired output is 3*x then we want the 
         output to say coeff, *, x and we want to coeff
         value to read 3, NaN, NaN. So, we want to calculated
         mse on 3 but not on the others.

NOTES:

TODO:
"""
import torch


def mse_loss_mask(predicted, target, use_index):
    out = (predicted[use_index]-target[use_index])**2
    return out.mean()


if __name__ == '__main__':
    import numpy as np

    predicted = torch.Tensor([[1, 2],
                              [3, 4]])
    target = torch.Tensor([[1, np.nan],
                           [np.nan, 4]])
    use_index = torch.where(~torch.isnan(target))

    loss = mse_loss_mask(predicted, target, use_index)
    print('loss', loss)
