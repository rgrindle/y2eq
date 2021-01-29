"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Script version of jupyter notebook of the same
         name.

NOTES: Only includes the training portion (no dataset generation)

TODO:
"""
from train import train
from evaluate import evaluate
from srvgd.architecture.torch.get_model import get_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import argparse
import random

import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    help='Provide name of model to continue training. '
                         'Example: if you can cnn_model1.pt and optimizer_model1.pt '
                         'use --checkpoint _model1 to continue training.')
args = parser.parse_args()

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def dataset_loader(train_dataset, test_dataset, batch_size=1024, valid_size=0.20):
    num_train = len(train_dataset)
    num_test_h = len(test_dataset)
    indices = list(range(num_train))
    test_idx_h = list(range(num_test_h))
    np.random.shuffle(test_idx_h)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=valid_sampler, num_workers=0)
    test_loader_h = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader_h, valid_idx, train_idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = torch.load(os.path.join('..', 'datasets', 'dataset_train_triple_points.pt'), map_location=device)
test_data = torch.load(os.path.join('..', 'datasets', 'dataset_test_triple_points.pt'), map_location=device)

print('train', len(train_data), len(train_data[0][0]), len(train_data[0][1]))
print('test', len(test_data), len(test_data[0][0]), len(test_data[0][1]))

train_loader, valid_loader, test_loader, valid_idx, train_idx = dataset_loader(train_data, test_data, batch_size=32, valid_size=0.30)

if args.checkpoint is None:
    model = get_model(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

else:
    print('Loading partly (or previously) trained model...', flush=True, end='')
    model = get_model(device, 'cnn{}.pt'.format(args.checkpoint))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(os.path.join('..', 'datasets', torch.load('optimizer{}.pt'.format(args.checkpoint)),
                                         map_location=device))
    print('done.')

print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.CrossEntropyLoss(ignore_index=0)
N_EPOCHS = 100
CLIP = 1

model = train(N_EPOCHS, train_loader, valid_loader,
              model, optimizer, criterion,
              CLIP, noise_Y=False, sigma=0.1)

test_loss = evaluate(model, test_loader, criterion)

print(f'| Test Loss: {test_loss:.3f} |')
