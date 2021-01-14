from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401
from eqlearner.dataset.processing.tokenization import get_string

import torch
import pandas as pd


eq_with_coeff = pd.read_csv('equations_with_coeff_train.csv', header=None).values
train_data = torch.load('dataset_train.pt')
eq = [t[1].tolist() for t in train_data]
y = [t[0].tolist() for t in train_data]
print(len(y), len(y[0]))

duplicates = []
for index, _ in enumerate(y):
    if y[index] in y[index+1:]:
        for i in range(index+1, len(y)):
            if y[index] == y[i]:
                print(get_string(eq[index]))
                print(get_string(eq[i]))
                print(eq_with_coeff[index])
                print(eq_with_coeff[i])
                print(y[index])
                print(y[i])
                print(index, i)
                break
        duplicates.append(index)
print(len(duplicates))
print(duplicates)
