"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 28, 2021

PURPOSE: Understand how untrained models behave.

NOTES:

TODO:
"""
from srvgd.utils.eval import translate_sentence
from srvgd.architecture.y2eq.get_y2eq_model import get_y2eq_model

import torch
import pandas as pd

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_y2eq_model(device,
                       layers=10)
test_data = torch.load(os.path.join('..', 'datasets', 'dataset_test_ff.pt'), map_location=device)

predicted_data = []
for i, obs in enumerate(test_data):

    inputs, targets = obs
    predicted = translate_sentence(sentence=inputs,
                                   model=model,
                                   device=device,
                                   max_len=67)[0]
    predicted_data.append(predicted)
    # print('.', flush=True, end='')
    print(i, predicted)

pd.DataFrame(predicted_data).to_csv('untrained_output.csv', index=False)
