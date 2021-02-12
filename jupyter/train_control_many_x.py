"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 4, 2021

PURPOSE: Script version of jupyter notebook of the same
         name.

NOTES: Only includes the training portion (no dataset generation)

TODO:
"""
from train_many_x import train
from srvgd.architecture.torch.get_model import get_model

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
parser.add_argument('--dataset', type=str,
                    help='Specify dataset to load and use for training. '
                         'it is assumed that there is dataset with the same name '
                         'except that train is replaced by test.',
                    default='dataset_train.pt')
parser.add_argument('--batch_size', type=int,
                    help='The batch size to use during training.',
                    default=2000)
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
parser.add_argument('--input_dim', type=int, choices=(1, 2), default=1,
                    help='The number of inputs in the dataset. '
                         'Indented for inputs include (x,y) or just '
                         'y.')
args = parser.parse_args()
print(args)

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


def put_x_in_dataset(x, dataset, device):
    inputs = [d[0].flatten().tolist() for d in dataset]
    outputs = [d[1].tolist() for d in dataset]
    new_inputs = []
    for i in inputs:
        new_inputs.append(np.vstack((x, i)).T.tolist())

    return TensorDataset(torch.Tensor(new_inputs).to(device), torch.LongTensor(outputs).to(device))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# train_data = torch.load('train_data_int_comp.pt')
train_data = torch.load(os.path.join('..', 'datasets', args.dataset), map_location=device)
# test_data = torch.load('test_data_int_comp.pt')
test_data = torch.load(os.path.join('..', 'datasets', args.dataset.replace('train', 'test')), map_location=device)

if 'with_x' not in args.dataset and 'with_1000x' not in args.dataset:
    x = (np.arange(0.1, 3.1, 0.1)-0.1)/3.0
    train_data = put_x_in_dataset(x, train_data, device)
    test_data = put_x_in_dataset(x, test_data, device)

print('train', len(train_data), len(train_data[0][0]))
print('test', len(test_data), len(test_data[0][0]))

train_loader, valid_loader, test_loader, valid_idx, train_idx = dataset_loader(train_data, test_data, batch_size=args.batch_size, valid_size=0.30)

if args.checkpoint is None:
    model = get_model(device,
                      INPUT_DIM=args.input_dim,
                      ENC_LAYERS=args.layers,
                      DEC_LAYERS=args.layers)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

else:
    print('Loading partly (or previously) trained model...', flush=True, end='')
    model = get_model(device,
                      path=os.path.join('..', 'models'),
                      load_weights='xy2eq{}.pt'.format(args.checkpoint),
                      ENC_LAYERS=args.layers,
                      DEC_LAYERS=args.layers,
                      INPUT_DIM=args.input_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(os.path.join('..', 'models', 'optimizer{}.pt'.format(args.checkpoint)),
                              map_location=device))
    print('done.')

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

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

# test_loss = evaluate(model, test_loader, criterion)

# print(f'| Test Loss: {test_loss:.3f} |')
