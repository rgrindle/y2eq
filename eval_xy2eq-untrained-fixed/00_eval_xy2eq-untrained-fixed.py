"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 19, 2021

PURPOSE: Get x and y to be used to evaluate xy2eq-untrained-fixed.

NOTES:

TODO:
"""
from srvgd.utils.eval import write_x_y_lists

write_x_y_lists('../datasets/equations_with_coeff_test_ff1000.csv',
                x_type='fixed')
