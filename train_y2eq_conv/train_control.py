"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 3, 2021

PURPOSE: Script version of jupyter notebook of the same
         name.

NOTES: Only includes the training portion (no dataset generation)

TODO:
"""
from srvgd.utils.mse_loss_mask import mse_loss_mask
from train import train
from srvgd.architecture.y2eq.get_y2eq_model import get_y2eq_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset
import numpy as np

import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    help='Provide name of model to continue training. '
                         'Example: if you can cnn_model1.pt and optimizer_model1.pt '
                         'use --checkpoint _model1 to continue training.')
parser.add_argument('--dataset', type=str, required=True,
                    help='Specify dataset to load and use for training. '
                         'it is assumed that there is dataset with the same name '
                         'except that train is replaced by test.',
                    default='dataset_train.pt')
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
parser.add_argument('--input_dim', type=int, choices=(1, 2), required=True,
                    help='The number of inputs in the dataset. '
                         'Indented for inputs include (x,y) or just '
                         'y.')
parser.add_argument('--include_coeffs', action='store_true',
                    help='If true, y2eq is expected to output '
                         'equations with coefficients not just '
                         'functional forms.')
parser.add_argument('--include_coeff_values', action='store_true',
                    help='If true, y2eq has an extra output unit '
                         'that is used to pick the coefficient value. '
                         '--include_coeffs must be used with the this cmd line arg')
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
    # num_test_h = len(test_dataset)
    indices = list(range(num_train))
    # test_idx_h = list(range(num_test_h))
    # np.random.shuffle(test_idx_h)
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
    # test_loader_h = DataLoader(test_dataset, batch_size=batch_size,
    # shuffle=False, num_workers=0)
    return train_loader, valid_loader, valid_idx, train_idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# train_data = torch.load('train_data_int_comp.pt')
train_data = torch.load(os.path.join('..', 'datasets', args.dataset), map_location=device)
# test_data = torch.load('test_data_int_comp.pt')
# test_data = torch.load(os.path.join('..', 'datasets', args.dataset.replace('train', 'test')), map_location=device)

print('train', len(train_data), len(train_data[0][0]), len(train_data[0][1]))
# print('test', len(test_data), len(test_data[0][0]), len(test_data[0][1]))

train_loader, valid_loader, valid_idx, train_idx = dataset_loader(train_data, batch_size=args.batch_size, valid_size=0.3)

print('len(train_loader)', len(train_loader))
print('len(valid_loader)', len(valid_loader))

if args.checkpoint is None:
    model = get_y2eq_model(device,
                           INPUT_DIM=args.input_dim,
                           ENC_LAYERS=args.layers,
                           DEC_LAYERS=args.layers,
                           DEC_MAX_LENGTH=len(train_data[0][1]),
                           include_coeffs=args.include_coeffs,
                           include_coeff_values=args.include_coeff_values)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

else:
    print('Loading partly (or previously) trained model...', flush=True, end='')
    model = get_y2eq_model(device,
                           path=os.path.join('..', 'models'),
                           load_weights='cnn{}.pt'.format(args.checkpoint),
                           INPUT_DIM=args.input_dim,
                           ENC_LAYERS=args.layers,
                           DEC_LAYERS=args.layers,
                           DEC_MAX_LENGTH=len(train_data[0][1]),
                           include_coeffs=args.include_coeffs,
                           include_coeff_values=args.include_coeff_values)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(os.path.join('..', 'models', 'optimizer{}.pt'.format(args.checkpoint)),
                              map_location=device))
    print('done.')

print(f'The model has {count_parameters(model):,} trainable parameters')

symbolic_loss = nn.CrossEntropyLoss(ignore_index=0)
coeff_loss = mse_loss_mask

if args.include_coeff_values:
    criterion = [symbolic_loss, coeff_loss]
else:
    criterion = symbolic_loss

epochs_filename = args.epochs

if args.checkpoint is not None:
    underscores_i = [i for i, s in enumerate(args.checkpoint) if s == '_']
    prev_epochs = int(args.checkpoint[underscores_i[-1]+1:])
    epochs_filename += prev_epochs


train(args.epochs, train_loader, valid_loader,
      model, optimizer, criterion,
      args.clip, noise_Y=False, sigma=0.1,
      save_loc=os.path.join('..', 'models'),
      save_end_name='_{}_batchsize{}_lr{}_clip{}_layers{}_includecoeffs{}_includecoeffvalues{}_{}'.format(args.dataset.replace('.pt', ''), args.batch_size, args.lr, args.clip, args.layers, args.include_coeffs, args.include_coeff_values, epochs_filename),
      with_x='with_x' in args.dataset,
      include_coeff_values=args.include_coeff_values)