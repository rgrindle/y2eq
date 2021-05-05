"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 23, 2021

PURPOSE: Get x and y to be used to evaluate y2eq-noa2-fixed-fixed.

NOTES:

TODO:
"""
from srvgd.utils.eval import write_x_y_lists

write_x_y_lists('../datasets/equations_with_coeff_test_ff1000_with_coeffs_v2.csv',
                x_type='fixed')
