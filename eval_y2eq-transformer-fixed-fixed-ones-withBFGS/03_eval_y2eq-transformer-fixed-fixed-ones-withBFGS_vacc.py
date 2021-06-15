"""
AUTHOR: Ryan Grindle

LAST MODFIFIED: May 19, 2021

PURPOSE: Group rmse files.

NOTES:

TODO:
"""
import pandas as pd

df = pd.read_csv('02_rmse_index0.csv')
for index in range(1, 971):
    df2 = pd.read_csv('02_rmse_index{}.csv'.format(index))
    df = df.append(df2, ignore_index=True)

print(df)
df.to_csv('02_rmse_150.csv', index=False)
