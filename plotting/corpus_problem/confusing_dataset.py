"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jul 13, 2021

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
data['normal dataset'] = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()
data['confusing dataset'] = pd.read_csv('../../eval_y2eq-transformer-confusing-fixed-fixed-noBFGS/02_rmse.csv')['rmse_int'].values.flatten()

plt.boxplot(data.values(), labels=data.keys())
plt.ylabel('Numeric cost on test dataset')
plt.yscale('log')
plt.savefig('confusing_dataset_pres.pdf')

results = mannwhitneyu(data['confusing dataset'], data['normal dataset'],
                       alternative='greater')
print(results)
