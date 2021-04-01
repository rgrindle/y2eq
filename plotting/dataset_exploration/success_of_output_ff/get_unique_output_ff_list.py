"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Get a list of the unique functional forms output
         by y2eq when given test dataset.

NOTES:

TODO:
"""
import numpy as np
import pandas as pd

y2eq_outputs = pd.read_csv('../../../eval_y2eq-fixed-fixed/01_predicted_ff.csv', header=None).values.flatten().tolist()
unique_y2eq_outputs = np.unique(y2eq_outputs)
pd.DataFrame(unique_y2eq_outputs).to_csv('unique_output_ff_list.csv',
                                         header=None,
                                         index=False)
