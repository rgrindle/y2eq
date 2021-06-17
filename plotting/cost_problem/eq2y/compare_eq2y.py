"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 16, 2021

PURPOSE: Compare different versions of eq2y. I think
         I know which is best, but is that just based
         on which plots I have looked at?

NOTES:

TODO:
"""
import pandas as pd
import matplotlib.pyplot as plt

more_ff = pd.read_csv('../../../eval_eq2y-transformer_more_ff/02_rmse.csv')['rmse_int'].values.flatten()
less_ff = pd.read_csv('../../../eval_eq2y-transformer/02_rmse.csv')['rmse_int'].values.flatten()
no_droput_more_ff = pd.read_csv('../../../eval_eq2y-transformer_no_dropout_more_ff/02_rmse.csv')['rmse_int'].values.flatten()
no_droput_less_ff = pd.read_csv('../../../eval_eq2y-transformer_no_dropout/02_rmse.csv')['rmse_int'].values.flatten()

print(more_ff.shape, less_ff.shape, no_droput_more_ff.shape)

plt.boxplot([more_ff, less_ff, no_droput_more_ff, no_droput_less_ff], labels=['7553 ff\'s', '1000 ff\'s', '7553 ff\'s\n(no dropout)', '100 ff\'s\n(no dropout)'])
plt.yscale('log')
plt.ylabel('RMSE on test dataset')
# plt.xlabel('Number of functional forms trained on')
plt.tight_layout()
plt.savefig('compare_eq2y.pdf')
