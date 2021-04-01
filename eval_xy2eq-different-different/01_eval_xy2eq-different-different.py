"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 27, 2021

PURPOSE: Confirm that fixed x values used during training
         is a problem for the current system. Step 1 (this
         script) is to evaluate the neural network on the
         updated y-values created in step 0.

NOTES: If NN outputs padding (0) only then the "equation"
       will be the empty set. The result will be the same
       if the NN ouputs END as the first token.

TODO:
"""
from srvgd.utils.normalize import normalize
from srvgd.utils.eval import translate_sentence
from srvgd.architecture.torch.get_model import get_model

import torch
import pandas as pd
import numpy as np

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device,
                  path='../models/',
                  load_weights='xy2eq_dataset_train_ff1000_with_1000x_batchsize2000_lr0.0001_clip1_layers10_900.pt',
                  INPUT_DIM=2)

with open('00_x_list.json', 'r') as json_file:
    x_list = json.load(json_file)

with open('00_y_normalized_list.json', 'r') as json_file:
    y_list = json.load(json_file)

x_min_ = 0.1
x_scale = 1./(3.1-0.1)

predicted_data = []
for i, (x, normalized_y) in enumerate(zip(x_list, y_list)):

    normalized_x = normalize(np.array(x), min_=x_min_, scale=x_scale)
    sentence = np.vstack((normalized_x, normalized_y)).T.tolist()

    predicted = translate_sentence(sentence=sentence,
                                   model=model,
                                   device=device,
                                   max_len=67)[0]
    predicted_data.append(predicted[5:-3])
    # print('.', flush=True, end='')
    print(i, predicted_data[-1])

pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)
