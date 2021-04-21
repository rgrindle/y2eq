from srvgd.architecture.plot2eq.models import Encoder, DecoderWithAttention
from srvgd.utils.get_data_image import get_data_image
from srvgd.updated_eqlearner.tokenization_rg import token_map, token_map_with_coeffs

import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import random_split

import os
import argparse

torch.set_num_threads(1)
torch.manual_seed(123)
# torch.set_deterministic(True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='Specify dataset to load and use for training. '
                         'it is assumed that there is dataset with the same name '
                         'except that train is replaced by test.',
                    default='dataset_train.pt')
parser.add_argument('--include_coeffs', action='store_true',
                    help='If true, y2eq is expected to output '
                         'equations with coefficients not just '
                         'functional forms.')
args = parser.parse_args()
print(args)

# Load dataset
dataset_path = os.path.join('..', '..', 'datasets')
dataset = torch.load(os.path.join(dataset_path, args.dataset))
# do 70%, 30% split
print(len(dataset))
train_dataset, val_dataset = random_split(dataset, [35000, 15000], generator=torch.Generator().manual_seed(42))

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 40  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_val_loss = float('inf')
print_freq = 100  # print training/validation stats every __ batches
# fine_tune_encoder = True  # fine-tune encoder?
# checkpoint = '../../models/checkpoint_1000x_20.pt'  # path to checkpoint, None if none
checkpoint = None
pretrained = False
resnet_num = 18
image_size = (64, 64)
assert resnet_num in (18, 34, 50, 101, 152)


def main():
    """
    Training and validation.
    """

    global best_val_loss, epochs_since_improvement, checkpoint, start_epoch, epochs

    if args.include_coeffs:
        vocab_size = len(token_map_with_coeffs)
    else:
        vocab_size = len(token_map)

    # Initialize / load checkpoint
    encoder = Encoder(resnet_num, pretrained).to(device)
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=vocab_size,
                                   encoder_dim=encoder.out_shape,
                                   dropout=dropout).to(device)

    if checkpoint is None:
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),
                                             lr=decoder_lr)
        encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        epochs += start_epoch
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_val_loss = checkpoint['val_loss']
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters())
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optimizer = torch.optim.Adam(params=encoder.parameters())
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)

        # Check if there was an improvement
        is_best = val_loss < best_val_loss
        best_val_loss = max(val_loss, best_val_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint('plot2eq_{}_resnet{}_pretrained{}_epochs{}'.format(args.dataset.replace('.pt', ''), resnet_num, pretrained, epochs), '../../models/', epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, val_loss, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (y, caps) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # make image
        x = torch.arange(0.1, 3.1, 0.1)[None]
        x = torch.repeat_interleave(x, len(y), dim=0)
        points = torch.stack((x, y[:, :, 0]), axis=-1)

        imgs = [np.flip(get_data_image(p, bins=image_size), axis=1) for p in points]
        imgs = torch.Tensor(imgs)

        caplens = torch.LongTensor([[c.count_nonzero()] for c in caps])

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    epoch_loss = 0

    # references = list()  # references (true captions) for calculating BLEU-4 score
    # hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (y, caps) in enumerate(val_loader):

            # make image
            x = torch.arange(0.1, 3.1, 0.1)[None]
            x = torch.repeat_interleave(x, len(y), dim=0)
            points = torch.stack((x, y[:, :, 0]), axis=-1)

            imgs = [np.flip(get_data_image(p, bins=image_size), axis=1) for p in points]
            imgs = torch.Tensor(imgs)

            caplens = torch.LongTensor([[c.count_nonzero()] for c in caps])

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            epoch_loss += loss.item()

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
    return losses.avg


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, path, epoch, epochs_since_improvement,
                    encoder, decoder, encoder_optimizer,
                    decoder_optimizer,
                    val_loss, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param best_val_loss: validation loss for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'val_loss': val_loss,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = 'checkpoint_' + data_name + '.pt'
    torch.save(state, os.path.join(path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(path, 'BEST_' + filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


if __name__ == '__main__':
    main()
