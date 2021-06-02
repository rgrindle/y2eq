"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Get x and y to be used to evaluate y2eq-no-coeffs-fixed-fixed.

NOTES:

TODO:
"""
from srvgd.utils.eval import write_x_y_lists

write_x_y_lists('../datasets/test_ff_list.csv',
                x_type='fixed')
