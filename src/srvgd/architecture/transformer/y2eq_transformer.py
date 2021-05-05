"""
AUTHOR: Ryan Grindle (modified from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer)

LAST MODIFIED: May 4, 2021

PURPOSE: Create a version of y2eq that is a
         transformer.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import token_map
from srvgd.architecture.transformer.utils import save_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

import os


class Transformer(nn.Module):
    def __init__(self,
                 embedding_size,
                 src_input_size,
                 trg_vocab_size,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 src_max_len,
                 trg_max_len,
                 device):
        super(Transformer, self).__init__()

        # src_word_linear replaced src_word_embedding because
        # input is floats.
        self.src_word_linear = nn.Linear(src_input_size, embedding_size)
        self.src_position_embedding = nn.Embedding(src_max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(trg_max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    def forward(self, src, trg):

        src_positions = (
            torch.arange(0, src.shape[1])
            .repeat(src.shape[0], 1)
            .to(self.device)
        )

        trg_seq_length = trg.shape[1]
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .repeat(trg.shape[0], 1)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_linear(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length-1).to(self.device)

        # embed_src.shape = [batch size, src seq length, feature number]
        # embed_trg.shape = [batch size, trg seq length, feature number]

        embed_src = embed_src.permute(1, 0, 2)
        embed_trg = embed_trg.permute(1, 0, 2)[:-1]
        # model expects shape = [sequence length, batch size, feature number]
        # so, thus the permute above

        out = self.transformer(
            embed_src,
            embed_trg,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


def get_nn_loss_batch(batch, model, device, criterion):

    inp_data = batch[0].to(device)
    target = batch[1].to(device)

    # Forward prop
    output = model(inp_data, target)

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


def train_one_epoch(train_iterator, model, device, criterion):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        print('.', flush=True, end='', sep='')
        optimizer.zero_grad()
        loss = get_nn_loss_batch(batch, model, device, criterion)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print('')
    mean_loss = sum(losses) / len(losses)
    return mean_loss


def valid_one_epoch(valid_iterator, model, device, criterion):
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(valid_iterator):
        print('.', flush=True, end='', sep='')
        loss = get_nn_loss_batch(batch, model, device, criterion)
        losses.append(loss.item())

    print('')
    mean_loss = sum(losses) / len(losses)
    return mean_loss


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

load_model = False

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_input_size = 1
trg_vocab_size = len(token_map)
embedding_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1
src_max_len = 30
trg_max_len = 100
forward_expansion = 1024

# Get dataset
dataset = torch.load(os.path.join('..', '..', '..', '..', 'datasets', 'dataset_train_ff1000.pt'),
                     map_location=device)
print('dataset', len(dataset), len(dataset[0][0]), len(dataset[0][1]))

train_dataset, valid_dataset = random_split(dataset, [35000, 15000], generator=torch.Generator().manual_seed(42))
print('train_dataset', len(train_dataset))
print('valid_dataset', len(valid_dataset))

train_iterator = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

valid_iterator = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=False)


model = Transformer(
    embedding_size,
    src_input_size,
    trg_vocab_size,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    src_max_len,
    trg_max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, factor=0.1, patience=10, verbose=True
# )

criterion = nn.CrossEntropyLoss(ignore_index=token_map[''])

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

train_losses = []
valid_losses = []
min_valid_loss = float('inf')
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    train_loss = train_one_epoch(train_iterator, model, device, criterion)
    train_losses.append(train_loss)

    valid_loss = valid_one_epoch(valid_iterator, model, device, criterion)
    valid_losses.append(valid_loss)

    print('train_loss:', train_loss)
    print('valid_loss:', valid_loss)

    checkpoint = {
        'train_loss': train_losses,
        'val_loss': valid_losses,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, '../../../../models/y2eq_transformer.pt')
    if valid_losses[-1] < min_valid_loss:
        save_checkpoint(checkpoint, '../../../../models/BEST_y2eq_transformer.pt')
        min_valid_loss = valid_losses[-1]
