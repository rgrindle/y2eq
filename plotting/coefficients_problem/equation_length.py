"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 14, 2021

PURPOSE: Are equation output by y2eq longer than expected?

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq, get_eq_string

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# get lengths of output equations
pred_ff_list = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/01_predicted_ff.csv').values.flatten()
print(pred_ff_list)

pred_tok_ff_list = [tokenize_eq(ff) for ff in pred_ff_list]
print(pred_tok_ff_list)

pred_tok_len_list = [len(ff) for ff in pred_tok_ff_list]
print(pred_tok_len_list)


# get lengths of true equations
dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
true_ff_list = np.unique([get_eq_string(d[1].tolist())[5:-3] for d in dataset])
true_tok_ff_list = [tokenize_eq(ff) for ff in true_ff_list]
print(true_ff_list)
print(true_tok_ff_list)
true_tok_len_list = [len(ff) for ff in true_tok_ff_list]
print(true_tok_len_list)

print(len(true_tok_ff_list), len(pred_tok_ff_list))

plt.boxplot([pred_tok_len_list, true_tok_len_list], labels=['pred', 'true'])
plt.ylabel('Number of tokens in functional form')
plt.savefig('equation_length.pdf')

results = mannwhitneyu(pred_tok_len_list, true_tok_len_list, alternative='greater')
print(results)
