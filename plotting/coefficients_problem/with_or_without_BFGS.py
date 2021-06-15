"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 14, 2021

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
data['no BFGS'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()
data['with BFGS'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-withBFGS/02_rmse_150.csv')['rmse_int'].values.flatten()

plt.boxplot(data.values(), labels=data.keys())
plt.ylabel('Normalized RMSE on interpolation region')
plt.yscale('log')
plt.savefig('with_or_without_BFGS.pdf')

results = mannwhitneyu(data['no BFGS'], data['with BFGS'],
                       alternative='greater')
print(results)
