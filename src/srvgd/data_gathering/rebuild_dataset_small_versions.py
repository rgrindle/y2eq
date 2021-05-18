"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 12, 2021

PURPOSE: Make a small version of dataset_train_ff1000.pt
         with only a few observations for sanity checks.

NOTES:

TODO:
"""
import torch
from torch.utils.data import TensorDataset

import os


num_observations = 3

dataset_path = os.path.join('..', '..', '..', 'datasets')

dataset = torch.load(os.path.join(dataset_path, 'dataset_train_ff1000.pt'),
                     map_location=torch.device('cpu'))

dataset_input = [d[0].tolist() for d in dataset][:num_observations]
dataset_output = [d[1].tolist() for d in dataset][:num_observations]

# pad dataset_output
max_len = max([len(out) for out in dataset_output])
padded_dataset_output = []
for i, out in enumerate(dataset_output):
    dataset_output[i] = out+[0]*(max_len-len(out))

print(len(dataset_input))
print(len(dataset_output))

dataset = TensorDataset(torch.Tensor(dataset_input), torch.LongTensor(dataset_output))
torch.save(dataset, os.path.join(dataset_path, 'dataset_train_ff1000_small{}.pt'.format(num_observations)))
