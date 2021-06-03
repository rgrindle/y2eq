"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 2, 2021

PURPOSE: Get x and y to be used to evaluate eq2y-transformer.

NOTES:

TODO:
"""
from srvgd.utils.eval import write_x_y_lists

write_x_y_lists('../datasets/equations_with_coeff_test_ff1000.csv',
                x_type='fixed')
