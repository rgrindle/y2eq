"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 15, 2021

PURPOSE: Compare y2eq with and without BFGS. I want to
         determine if y2eq is actually good at determining
         the correct functional form.

NOTES:

TODO:
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

data = {}
data['y2eq no BFGS'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()
data['y2eq-pad no BFGS'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS-pad/02_rmse.csv')['rmse_int'].values.flatten()
data['y2eq-pad with BFGS'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-withBFGS-pad/02_rmse_150.csv')['rmse_int'].values.flatten()

plt.boxplot(data.values(), labels=data.keys())
plt.ylabel('Normalized RMSE on interpolation region')
plt.yscale('log')
plt.savefig('with_or_without_BFGS_pad.pdf')

results = mannwhitneyu(data['y2eq-pad no BFGS'],
                       data['y2eq-pad with BFGS'],
                       alternative='greater')
print('y2eq-pad no BFGS > y2eq-pad with BFGS', results)


results = mannwhitneyu(data['y2eq no BFGS'],
                       data['y2eq-pad no BFGS'],
                       alternative='two-sided')
print('y2eq no BFGS < y2eq-pad no BFGS', results)
