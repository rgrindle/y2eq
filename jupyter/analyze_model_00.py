"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 5, 2020

PURPOSE: Get output of trained NN on test dataset.

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, 'cnn_epochs100.pt')

test_data = torch.load('test_data_int_comp.pt', map_location=device)
test_data = DataLoader(test_data, batch_size=1000)
output_data = []
with torch.no_grad():
    for batch in test_data:
        print('batch')
        src = batch[0]
        trg = batch[1]
        output, _ = model(src, trg[:, :-1])
        output_data.append([np.argmax(o.cpu().numpy(), axis=1) for o in output])

output_data = np.array(list(itertools.chain(*output_data)))
print(output_data.shape)
pd.DataFrame(output_data).to_csv('test_output.csv', index=False)
