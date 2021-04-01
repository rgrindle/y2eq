"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 30, 2021

PURPOSE: Confirm that fixed x values used during training
         is a problem for the current system. Step 1 (this
         script) is to evaluate the neural network on the
         updated y-values created in step 0.

NOTES: If NN outputs padding (0) only then the "equation"
       will be the empty set. The result will be the same
       if the NN ouputs END as the first token.

TODO:
"""
from srvgd.utils.eval import translate_sentence
from srvgd.architecture.torch.get_model import get_model

import torch
import pandas as pd

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_endname = '_dataset_train_ff1000_2d_batchsize32_lr0.0001_clip1_layers10_105'
model = get_model(device,
                  path='../models/',
                  OUTPUT_DIM=26,
                  ENC_MAX_LENGTH=1024,
                  DEC_MAX_LENGTH=65,
                  load_weights='y2eq_2d{}.pt'.format(file_endname))

with open('00_y_normalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)


predicted_data = []
for i, y in enumerate(y_list):

    predicted = translate_sentence(sentence=y,
                                   model=model,
                                   device=device,
                                   max_len=67,
                                   two_d=True)[0]
    predicted_data.append(predicted[5:-3])
    # print('.', flush=True, end='')
    print(i, predicted_data[-1])

pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)
