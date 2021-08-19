from evaluate import evaluate

import torch
import numpy as np
import pandas as pd

import time
import os


def train(num_epochs, train_loader, valid_loader,
          model, optimizer, criterion,
          clip, noise_Y=False, sigma=0.1,
          save_loc='', save_end_name='',
          with_x=False, include_coeff_values=False):
    best_valid_loss = float('inf')
    torch.set_num_threads(1)
    history = {'train_loss': [], 'valid_loss': []}
    if save_loc != '':
        os.makedirs(save_loc, exist_ok=True)

    for epoch in range(num_epochs):

        start_time = time.time()

        if include_coeff_values:
            train_loss = train_one_epoch_coeff_values(model, train_loader, optimizer,
                                                      criterion, clip, noise_Y=False,
                                                      sigma=0.05, with_x=with_x)
            valid_loss = evaluate_coeff_values(model, valid_loader, criterion, with_x=with_x)
        else:
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         criterion, clip, noise_Y=False,
                                         sigma=0.05, with_x=with_x)
            valid_loss = evaluate(model, valid_loader, criterion, with_x=with_x)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_loc, 'cnn{}.pt'.format(save_end_name)))
            torch.save(optimizer.state_dict(), os.path.join(save_loc, 'optimizer{}.pt'.format(save_end_name)))
            print('saving')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f})')
        print(f'\t Val. Loss: {valid_loss:.3f})')
        print('optimizer learning rate', optimizer.param_groups[0]['lr'])

    pd.DataFrame(history.values()).T.to_csv(os.path.join(save_loc, 'train_history{}.csv'.format(save_end_name)), index=False, header=list(history.keys()))
    # model.load_state_dict(torch.load('cnn.pt'))
    # return model


def train_one_epoch(model, iterator, optimizer, criterion,
                    clip, noise_Y=False, sigma=0.1, with_x=False):
    model.train()

    epoch_loss = 0
    print('number of batches = ', len(iterator))

    for i, batch in enumerate(iterator):
        print('.', end='', flush=True)
        if noise_Y:
            src = batch[0] + torch.from_numpy(sigma*np.random.randn(batch[0].shape[0], 30)).float().cuda()
            trg = batch[1]
        else:
            src = batch[0]
            trg = batch[1]

        if with_x:
            # pick 30 random points
            # and remove x
            print(src.shape)
            src = src[:, :, 1:]
            print(src.shape)

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        # 1: removes start of sequence token
        trg = trg[:, 1:].contiguous().view(-1)  # view = reshape

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    print('')
    return epoch_loss / len(iterator)


def train_one_epoch_coeff_values(model, iterator, optimizer, criterion,
                                 clip, noise_Y=False, sigma=0.1, with_x=False):
    model.train()

    epoch_loss = 0
    print('number of batches = ', len(iterator))

    for i, batch in enumerate(iterator):
        print('.', end='', flush=True)
        optimizer.zero_grad()
        loss = get_nn_loss_coeff_values(batch, model, criterion, with_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    print('')
    return epoch_loss / len(iterator)


def evaluate_coeff_values(model, iterator, criterion, with_x=False):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            loss = get_nn_loss_coeff_values(batch, model, criterion, with_x)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def get_nn_loss_coeff_values(batch, model, criterion, with_x):
    src = batch[0]
    token_trg = batch[1]
    coeff_trg = batch[2]

    if with_x:
        # pick 30 random points
        # and remove x
        print(src.shape)
        src = src[:, :, 1:]
        print(src.shape)

    token_output, _, coeff_output = model(src, token_trg[:, :-1])

    coeff_output = coeff_output.squeeze(-1)
    # token_output = [batch size, trg len - 1, output dim]
    # token_trg = [batch size, trg len]

    output_dim = token_output.shape[-1]
    token_output = token_output.contiguous().view(-1, output_dim)
    # 1: removes start of sequence token
    token_trg = token_trg[:, 1:].contiguous().view(-1)  # view = reshape

    # token_output = [batch size * token_trg len - 1, output dim]
    # token_trg = [batch size * token_trg len - 1]

    coeff_use_index = torch.where(~torch.isnan(coeff_trg))
    token_loss = criterion[0](token_output, token_trg)
    coeff_loss = criterion[1](coeff_output, coeff_trg, coeff_use_index)
    loss = token_loss + coeff_loss
    return loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
