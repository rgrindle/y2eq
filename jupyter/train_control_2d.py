"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 16, 2021

PURPOSE: Train y2eq on 2D equations.

NOTES: Uses train_2d.py and evaluate_2d.py

TODO:
"""
from train_2d import train
from srvgd.architecture.torch.get_model import get_model
from srvgd.updated_eqlearner.tokenization_rg import token_map_2d

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
parser.add_argument('--dataset', type=str,
                    help='Specify dataset to load and use for training. ',
                    default='dataset_train_ff1000_2d.pt')
parser.add_argument('--batch_size', type=int,
                    help='The batch size to use during training.',
                    default=32)
parser.add_argument('--lr', type=float,
                    help='Learning rate used by Adam',
                    default=0.0001)
parser.add_argument('--clip', type=float,
                    help='Value for gradient clipping.',
                    default=1)
parser.add_argument('--layers', type=int,
                    help='Number of layers in encoder/decoder',
                    default=10)
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train for.',
                    default=100)

args = parser.parse_args()
print(args)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def dataset_loader(train_dataset, batch_size=1024, valid_size=0.20):
    num_train = len(train_dataset)
    indices = list(range(num_train))
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
    return train_loader, valid_loader, valid_idx, train_idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

assert '2d' in args.dataset

train_data = torch.load(os.path.join('..', 'datasets', args.dataset), map_location=device)

print('train', len(train_data), len(train_data[0][0]), len(train_data[0][0][0]), len(train_data[0][1]))

train_loader, valid_loader, valid_idx, train_idx = dataset_loader(train_data, batch_size=args.batch_size, valid_size=0.30)

eq_max_length = len(train_data[0][1])
num_y_values = 1024

if args.checkpoint is None:
    model = get_model(device,
                      INPUT_DIM=1,
                      ENC_LAYERS=args.layers,
                      DEC_LAYERS=args.layers,
                      OUTPUT_DIM=len(token_map_2d()),
                      ENC_MAX_LENGTH=num_y_values,
                      DEC_MAX_LENGTH=eq_max_length)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

else:
    print('Loading partly (or previously) trained model...', flush=True, end='')
    model = get_model(device,
                      path=os.path.join('..', 'models'),
                      load_weights='y2eq_2d{}.pt'.format(args.checkpoint),
                      ENC_LAYERS=args.layers,
                      DEC_LAYERS=args.layers,
                      OUTPUT_DIM=len(token_map_2d()),
                      ENC_MAX_LENGTH=num_y_values,
                      DEC_MAX_LENGTH=eq_max_length)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(os.path.join('..', 'models', 'y2eq_2d_optimizer{}.pt'.format(args.checkpoint)),
                              map_location=device))
    print('done.')

print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.CrossEntropyLoss(ignore_index=0)

epochs_filename = args.epochs

if args.checkpoint is not None:
    underscores_i = [i for i, s in enumerate(args.checkpoint) if s == '_']
    prev_epochs = int(args.checkpoint[underscores_i[-1]+1:])
    epochs_filename += prev_epochs


train(args.epochs, train_loader, valid_loader,
      model, optimizer, criterion,
      args.clip, noise_Y=False, sigma=0.1,
      save_loc=os.path.join('..', 'models'),
      save_end_name='_{}_batchsize{}_lr{}_clip{}_layers{}_{}'.format(args.dataset.replace('.pt', ''), args.batch_size, args.lr, args.clip, args.layers, epochs_filename))
