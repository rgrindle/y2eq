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
from srvgd.utils.eval import translate_sentence, normalize
from srvgd.architecture.torch.get_model import get_model

import torch
import pandas as pd
import numpy as np

import os
import json


# init model as if for a xy2eq ...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, INPUT_DIM=2)

# but get y2eq weights ...
path = '../models/'
load_weights = 'cnn_dataset_train_ff1000_batchsize32_lr0.0001_clip1_layers10_105.pt'

# and fill missing weights (weights for x) with 0's
weights = torch.load(os.path.join(path, load_weights), map_location=device)
weights['encoder.tok_linear.weight'] = torch.hstack([torch.zeros((256, 1)), weights['encoder.tok_linear.weight']])
model.load_state_dict(weights)


def test_weight_choice():
    # Make some fake data, where
    # the only difference is the x
    # either (x1 or x2).
    f = lambda x: x**2 + x
    x1 = np.linspace(0, 1, 30)
    x2 = 0.5*np.ones(30)
    y = normalize(f(x1))

    s1 = np.vstack((x1, y)).T.tolist()
    s2 = np.vstack((x2, y)).T.tolist()

    # Get two output sentences and compare
    predicted1 = translate_sentence(sentence=s1,
                                    model=model,
                                    device=device,
                                    max_len=67)[0]
    print(predicted1[5:-3])

    predicted2 = translate_sentence(sentence=s2,
                                    model=model,
                                    device=device,
                                    max_len=67)[0]
    print(predicted2[5:-3])

    assert predicted1 == predicted1


# test_weight_choice()

# Now get output of xy2eq using y2eq weights
# We should get exactly the same result. As using
# y2eq now.
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

pd.DataFrame(predicted_data).to_csv('01_predicted_ff_weights.csv', index=False, header=None)
