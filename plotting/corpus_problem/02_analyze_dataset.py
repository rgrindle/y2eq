"""
AUTHOR: Ryan Grindle

LAST MODIFIED: June 15, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES:

TODO:
"""
import numpy as np
import pandas as pd

error_mat = pd.read_csv('error_mat.csv', index=False, header=None).values
print(error_mat.shape)

assert np.all(error_mat == error_mat.T), 'error_mat is not symmetric! problem in previous script'

# Since error_mat is symmetric and we don't really care
# that the diagonal is all zeros, we set the upper triangular
# part of error_mat to inf (including the diagonal). This
# way we don't have to deal with any duplicate comparisons.
inf_indices = np.triu_indices(50000)
error_mat[inf_indices] = np.inf

sorted_index = np.argsort(error_mat)
unrav_sorted_index = np.unravel_index(sorted_index)
print(unrav_sorted_index[:10])
print(error_mat[unrav_sorted_index[:10]])

pd.DataFrame(unrav_sorted_index).to_csv('unraveled_sorted_index.csv',
                                        index=False,
                                        header=None)
