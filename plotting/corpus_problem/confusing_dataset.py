"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 29, 2021

PURPOSE: Compare y2eq (without BFGS) when trained on
         the normal dataest and when trained on one
         generated that has semantically similar instances
         of functional forms.

NOTES:

TODO:
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

data = {}
data['normal'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()
data['confusing'] = pd.read_csv('../../eval_y2eq-transformer-confusing-fixed-fixed-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()

plt.boxplot(data.values(), labels=data.keys())
plt.ylabel('Normalized RMSE on interpolation region')
plt.yscale('log')
plt.savefig('confusing_dataset.pdf')

results = mannwhitneyu(data['confusing'], data['normal'],
                       alternative='greater')
print(results)
