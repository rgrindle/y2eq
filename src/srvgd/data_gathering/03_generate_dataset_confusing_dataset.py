"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 21, 2021

PURPOSE: Take a list of functional forms and generate
         a dataset of a given size from them. Inputs will
         be y-values and outputs will be the functional forms.
         I expect that examples of different functional forms
         that result in similar y-values will be more difficult
         for the NN to learn from. To test this, I am making
         a dataset that has many similar y-values.

NOTES: In 03 (this script), join all the pieces of the dataset
       that have been created by 01 and 02.

TODO:
"""
import torch
from torch.utils.data import TensorDataset
import pandas as pd

import os
import json


eq_list = []
dataset_input_list = []
dataset_output_list = []
for job_num in range(750):
    with open('confusing_dataset_in_pieces/_train_ff1000_confusing_jobnum{}.json'.format(job_num)) as f:
        data = json.load(f)
        eq_list.extend(data['eq_list'])
        dataset_input_list.extend(data['input'])
        dataset_output_list.extend(data['output'])

print(len(eq_list), len(dataset_input_list), len(dataset_output_list))

dataset_path = os.path.join('..', '..', '..', 'datasets')

fourth_dataset_name = '_train_ff1000_confusing_fourth'

eq_list_fourth = pd.read_csv(os.path.join(dataset_path,
                                          'equations_with_coeff'+fourth_dataset_name+'.csv'),
                             header=None).values.flatten()
print(eq_list_fourth.shape)

fourth_dataset = torch.load(os.path.join(dataset_path, 'dataset'+fourth_dataset_name+'.pt'))
dataset_input_fourth_list = [d[0].tolist() for d in fourth_dataset]
dataset_output_fourth_list = [d[1].tolist() for d in fourth_dataset]

print(len(dataset_input_fourth_list))
print(len(dataset_output_fourth_list))

eq_list.extend(eq_list_fourth)
dataset_input_list.extend(dataset_input_fourth_list)
dataset_output_list.extend(dataset_output_fourth_list)

print(len(eq_list))
print(len(dataset_input_list))
print(len(dataset_output_list))

# pad output
max_len = max([len(out) for out in dataset_output_list])
for i, out in enumerate(dataset_output_list):
    dataset_output_list[i] = out+[0]*(max_len-len(out))

full_dataset = TensorDataset(torch.Tensor(dataset_input_list).unsqueeze(-1),
                             torch.LongTensor(dataset_output_list))
torch.save(full_dataset, os.path.join(dataset_path, 'dataset_train_ff1000_confusing.pt'))
