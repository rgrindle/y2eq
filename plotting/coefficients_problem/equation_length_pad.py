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


def get_ff_lengths(ff_list):
    tok_ff_list = [tokenize_eq(ff) for ff in ff_list]
    tok_len_list = [len(ff) for ff in tok_ff_list]
    return tok_len_list


# get lengths of output equations
pred_ff_list = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS/01_predicted_ff.csv').values.flatten()
pred_tok_len_list = get_ff_lengths(pred_ff_list)

pred_pad_ff_list = pd.read_csv('../../eval_y2eq-transformer-fixed-fixed-ones-noBFGS-pad/01_predicted_ff.csv').values.flatten()
pred_pad_tok_len_list = get_ff_lengths(pred_pad_ff_list)

# get lengths of true equations
dataset = torch.load('../../datasets/dataset_train_ff1000.pt')
true_ff_list = np.unique([get_eq_string(d[1].tolist())[5:-3] for d in dataset])
true_tok_len_list = get_ff_lengths(true_ff_list)

print(len(true_tok_len_list), len(pred_tok_len_list), len(pred_pad_tok_len_list))

plt.boxplot([pred_tok_len_list, pred_pad_tok_len_list, true_tok_len_list],
            labels=['y2eq', 'y2eq-pad', 'true'])
plt.ylabel('Number of tokens in functional form')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('equation_length_pad.pdf')

results = mannwhitneyu(true_tok_len_list, pred_pad_tok_len_list, alternative='less')
print('true tok len < y2eq-pad tok len:', results)

results = mannwhitneyu(pred_pad_tok_len_list, pred_tok_len_list, alternative='less')
print('y2eq-pad tok len < y2eq tok len:', results)
