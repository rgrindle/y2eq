"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 6, 2021

PURPOSE: Get output of trained NN on train dataset.

NOTES:

TODO:
"""
from get_model import get_model
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import itertools


file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates'
# file_endname = '_epochs100_0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, 'cnn{}.pt'.format(file_endname))
model.eval()

train_data = torch.load('dataset_train.pt', map_location=device)
train_data = DataLoader(train_data, batch_size=2000)
output_data = []
with torch.no_grad():
    for batch in train_data:
        print('batch')
        src = batch[0]
        trg = batch[1]
        output, _ = model(src, trg[:, :-1])
        output_data.append(np.argmax(output.cpu().numpy(), axis=2))
        assert np.max(output_data) <= 22
        assert np.min(output_data) >= 0
output_data = np.array(list(itertools.chain(*output_data)))
print(output_data.shape)
pd.DataFrame(output_data).to_csv('train_output{}.csv'.format(file_endname), index=False)
