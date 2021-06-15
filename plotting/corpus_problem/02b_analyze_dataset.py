"""
AUTHOR: Ryan Grindle

LAST MODIFIED: June 15, 2021

PURPOSE: Can I find examples of equations that already
         exist in my current dataset that are similar?

NOTES:

TODO:
"""
import pandas as pd
import matplotlib.pyplot as plt

error_mat = pd.read_csv('error_mat.csv', index=False, header=None).values
print(error_mat.shape)

plt.imshow(error_mat, cmap='Greys')
cbar = plt.colorbar()
cbar.set_label('RMSE')
plt.savefig('error_mat_heatmap.png')
