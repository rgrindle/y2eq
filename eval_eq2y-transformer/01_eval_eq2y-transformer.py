"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 2, 2021

PURPOSE: Evaluate y2eq-transformer-fixed-fixed and save functional form.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.eq2y import eq2y_trans_model
from srvgd.updated_eqlearner.tokenization_rg import tokenize_eq

import torch
import pandas as pd

import json

ff_list = pd.read_csv('../datasets/test_ff_list.csv', header=None).values.flatten()
ff_list = [tokenize_eq(ff) for ff in ff_list]

# pad
max_len = max([len(ff) for ff in ff_list])
for i, ff in enumerate(ff_list):
    ff_list[i] = ff+[0]*(max_len-len(ff))

ff_list = torch.LongTensor(ff_list)

print(ff_list.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dict = torch.load('../models/BEST_eq2y_transformer_3300.pt',
                        map_location=device)
eq2y_trans_model.load_state_dict(model_dict['state_dict'])

predicted_y = eq2y_trans_model(ff_list).permute(1, 0, 2)
print(predicted_y.shape)

with open('01_predicted_y.json', 'w') as file:
    json.dump(predicted_y.tolist(), file)
