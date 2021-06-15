"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 14, 2021

PURPOSE: This file contains functions for training
         a transformer model and measuring loss on
         the validation dataset.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from srvgd.updated_eqlearner.tokenization_rg import token_map

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

import os


def get_nn_loss_batch(batch, model, device, criterion):

    inp_data = batch[0].to(device)
    target = batch[1].to(device)

    # Forward prop
    output = model(inp_data, target)

    pred_token_indices = output.permute(1, 0, 2).argmax(2).tolist()
    for i, raw_eq in enumerate(pred_token_indices):
        pred_ff = get_eq_string(raw_eq)
        print(i, pred_ff)

    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * batch_size that we want to send in into
    # our cost function, so we need to do some reshaping.
    # Let's also remove the start token while we're at it
    output = output.reshape(-1, output.shape[2])
    target = target.permute(1, 0)[1:].reshape(-1)

    loss = criterion(output, target)
    return loss


def train_one_epoch(train_iterator, model, device, criterion, optimizer,
                    get_nn_loss_batch_=get_nn_loss_batch):
    print('TRAINING')
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        print('.', flush=True, end='', sep='')
        optimizer.zero_grad()
        loss = get_nn_loss_batch_(batch, model, device, criterion)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print('')
    mean_loss = sum(losses) / len(losses)
    return mean_loss


def valid_one_epoch(valid_iterator, model, device, criterion,
                    get_nn_loss_batch_=get_nn_loss_batch):
    print('EVALUATING')
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(valid_iterator):
        print('.', flush=True, end='', sep='')
        loss = get_nn_loss_batch_(batch, model, device, criterion)
        losses.append(loss.item())

    print('')
    mean_loss = sum(losses) / len(losses)
    return mean_loss


def save_checkpoint(state, filename):
    print('=> Saving checkpoint at '+filename)
    torch.save(state, filename)


def get_dataset(dataset_name, device):
    dataset = torch.load(os.path.join('..', '..', '..', '..', 'datasets', dataset_name),
                         map_location=device)
    print('dataset', len(dataset), len(dataset[0][0]), len(dataset[0][1]))
    return dataset


def split_dataset(dataset, split=(35000, 15000), batch_size=32):
    train_dataset, valid_dataset = random_split(dataset, split, generator=torch.Generator().manual_seed(42))
    print('train_dataset', len(train_dataset))
    print('valid_dataset', len(valid_dataset))

    train_iterator = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    valid_iterator = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return train_iterator, valid_iterator


def train_many_epochs(train_iterator, valid_iterator,
                      model, device, model_name,
                      learning_rate=3e-4, num_epochs=100,
                      criterion=nn.CrossEntropyLoss(ignore_index=token_map['']),
                      get_nn_loss_batch_=get_nn_loss_batch,
                      train_losses=None, valid_losses=None,
                      optimizer_state_dict=None):
    assert model_name[-3:] == '.pt'

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    if train_losses is None:
        train_losses = []

    if valid_losses is None:
        valid_losses = []
        min_valid_loss = float('inf')
    else:
        min_valid_loss = min(valid_losses)

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        train_loss = train_one_epoch(train_iterator, model, device, criterion, optimizer, get_nn_loss_batch_)
        train_losses.append(train_loss)

        valid_loss = valid_one_epoch(valid_iterator, model, device, criterion, get_nn_loss_batch_)
        valid_losses.append(valid_loss)

        print('train_loss:', train_loss)
        print('valid_loss:', valid_loss)

        checkpoint = {
            'train_loss': train_losses,
            'val_loss': valid_losses,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, '../../../../models/'+model_name)
        if valid_losses[-1] < min_valid_loss:
            save_checkpoint(checkpoint, '../../../../models/BEST_'+model_name)
            min_valid_loss = valid_losses[-1]
