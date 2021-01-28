"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 22, 2021

PURPOSE: Get output of trained NN on train dataset.

NOTES:

TODO:
"""
from srvgd.architecture.torch.get_model import get_model
from srvgd.updated_eqlearner.tensor_dataset_rg import TensorDatasetCPU as TensorDataset  # noqa: F401
from srvgd.utils.eval import translate_sentence

import torch
import pandas as pd

import os

file_endname = '_layers10_clip1_dropoutFalse_lr1e-4_2000'
# file_endname = '_epochs100_0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, 'cnn{}.pt'.format(file_endname))
model.eval()

# train_data = torch.load('dataset_train.pt', map_location=device)
train_data = torch.load(os.path.join('..', 'datasets', 'train_data_int_comp.pt'), map_location=device)

predicted_data = []
for obs in train_data:
    inputs, targets = obs
    predicted = translate_sentence(sentence=inputs,
                                   model=model,
                                   device=device)[0]
    predicted_data.append(predicted)
    print('.', flush=True, end='')

pd.DataFrame(predicted_data).to_csv('train_output{}.csv'.format(file_endname), index=False)
