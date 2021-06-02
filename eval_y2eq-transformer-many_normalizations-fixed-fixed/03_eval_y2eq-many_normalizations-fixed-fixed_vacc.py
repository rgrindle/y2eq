"""
AUTHOR: Ryan Grindle

LAST MODFIFIED: Jun 1, 2021

PURPOSE: Group rmse files.

NOTES:

TODO:
"""
import pandas as pd

df = pd.read_csv('02_rmse_index0.csv')
for index in range(1, 1000):
    df2 = pd.read_csv('02_rmse_index{}.csv'.format(index))
    df = df.append(df2, ignore_index=True)

print(df)
df.to_csv('02_rmse_1000.csv', index=False)
