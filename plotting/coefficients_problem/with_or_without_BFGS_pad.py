"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jul 13, 2021

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
data['y2eq'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()
data['y2eq-pad'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS-pad/02_rmse.csv')['rmse_int'].values.flatten()
# data['y2eq-pad with nonlinear regression'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-withBFGS-pad/02_rmse_150.csv')['rmse_int'].values.flatten()

plt.boxplot(data.values(), labels=data.keys())
plt.ylabel('Numeric cost on test dataset')
plt.yscale('log')
plt.savefig('with_or_without_BFGS_pad_pres.pdf')

# results = mannwhitneyu(data['y2eq-pad no L-BFGS-B'],
#                        data['y2eq-pad with L-BFGS-B'],
#                        alternative='greater')
# print('y2eq-pad no BFGS > y2eq-pad with BFGS', results)


results = mannwhitneyu(data['y2eq'],
                       data['y2eq-pad'],
                       alternative='two-sided')
print('y2eq no BFGS < y2eq-pad no BFGS', results)
