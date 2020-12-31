from evaluate import evaluate

import torch
import numpy as np
import pandas as pd

import time
import math


def train(num_epochs, train_loader, valid_loader,
          model, optimizer, criterion,
          clip, noise_Y=False, sigma=0.1):
    best_valid_loss = float('inf')
    torch.set_num_threads(1)
    history = {'train_loss': [], 'valid_loss': []}

    for epoch in range(num_epochs):

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, clip, noise_Y=False,
                                     sigma=0.05)
        valid_loss = evaluate(model, valid_loader, criterion)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'cnn.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}) | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}) |  Val. PPL: {math.exp(valid_loss):7.3f}')

    pd.DataFrame(history.values()).T.to_csv('train_history.csv', index=False, header=list(history.keys()))
    model.load_state_dict(torch.load('cnn.pt'))
    return model


def train_one_epoch(model, iterator, optimizer, criterion,
                    clip, noise_Y=False, sigma=0.1):
    model.train()

    epoch_loss = 0
    print('batch_size = ', len(iterator))

    for i, batch in enumerate(iterator):
        print('.', end='', flush=True)
        if noise_Y:
            src = batch[0] + torch.from_numpy(sigma*np.random.randn(batch[0].shape[0], 30)).float().cuda()
            trg = batch[1]
        else:
            src = batch[0]
            trg = batch[1]

        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    print('')
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
