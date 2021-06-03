"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jun 1, 2021

PURPOSE: Train y2eq with symbolic and numeric
         loss. Requires using a pre-trained eq2y
         to get a differentiable path from eq to y.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.train_and_validate import get_dataset, split_dataset, train_many_epochs
from srvgd.architecture.transformer.y2eq_transformer import y2eqTransformer
from srvgd.architecture.transformer.eq2y import eq2yTransformer
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from srvgd.updated_eqlearner.tokenization_rg import token_map
from equation.EquationInfix import EquationInfix

import torch
import torch.nn as nn

import os
import argparse


class y2eqTwoLosses(nn.Module):
    def __init__(self,
                 numeric_loss_weight,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(y2eqTwoLosses, self).__init__()

        self.numeric_loss_weight = numeric_loss_weight
        self.device = device

        self.y2eq = y2eqTransformer().to(self.device)
        self.eq2y = eq2yTransformer().to(self.device)

        eq2y_checkpoint = torch.load(os.path.join('../../../../models/BEST_eq2y_transformer.pt'),
                                     map_location=self.device)
        self.eq2y.load_state_dict(eq2y_checkpoint['state_dict'])

        # Don't change weights of eq2y
        for parameter in self.eq2y.parameters():
            parameter.requires_grad = False

    def forward(self, y_true, eq_true):
        eq_pred = self.y2eq(y_true, eq_true)

        y_pred = None
        if self.numeric_loss_weight > 0.:
            eq_pred_token_indices = eq_pred.permute(1, 0, 2).argmax(2)
            y_pred = self.eq2y(eq_pred_token_indices)

        return eq_pred, y_pred


def get_model(numeric_loss_weight,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    y2eq_model_two_losses = y2eqTwoLosses(numeric_loss_weight).to(device)
    return y2eq_model_two_losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numeric_loss_weight', type=float, required=True,
                        help='weight of numeric loss. symbolic_loss_weight '
                             'will be 1 - numeric_loss_weight')
    args = parser.parse_args()

    def get_nn_loss_batch_(batch, model, device, criterion):
        """Note that criterion is irrelevant for this
        function."""

        symbolic_criterion = nn.CrossEntropyLoss(ignore_index=token_map[''])
        numeric_criterion = torch.nn.MSELoss()

        y_true = batch[0].to(device)
        eq_true = batch[1].to(device)

        # Forward prop
        eq_pred, y_pred = model(y_true, eq_true)

        pred_token_indices = eq_pred.permute(1, 0, 2).argmax(2).tolist()
        eq_valid_index_list = []
        for i, raw_eq in enumerate(pred_token_indices):
            pred_eq_str = get_eq_string(raw_eq)
            eq = EquationInfix(pred_eq_str, apply_coeffs=False)
            if eq.is_valid():
                eq_valid_index_list.append(i)
            print(i, pred_eq_str)

        # eq_pred is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping.
        # Let's also remove the start token while we're at it
        eq_pred = eq_pred.reshape(-1, eq_pred.shape[2])
        eq_true = eq_true.permute(1, 0)[1:].reshape(-1)

        symbolic_loss = symbolic_criterion(eq_pred, eq_true)

        if len(eq_valid_index_list) > 0:
            eq_valid_index_list = torch.LongTensor(eq_valid_index_list)
            y_pred = y_pred.permute(1, 0, 2)
            numeric_loss = numeric_criterion(y_pred[eq_valid_index_list], y_true[eq_valid_index_list])
            loss = args.numeric_loss_weight*numeric_loss + (1-args.numeric_loss_weight)*symbolic_loss
        else:
            print('symbolic loss only')
            loss = symbolic_loss

        return loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y2eq_model_two_losses = get_model(args.numeric_loss_weight, device)

    # Get number of trainable parameters
    num_params = sum(p.numel() for p in y2eq_model_two_losses.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    dataset_name = 'dataset_train_ff1000_no_coeffs.pt'
    dataset = get_dataset(dataset_name, device)
    train_iterator, valid_iterator = split_dataset(dataset)

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=y2eq_model_two_losses,
                      device=device,
                      model_name='y2eq_two_losses_numeric_weight{}.pt'.format(args.numeric_loss_weight),
                      get_nn_loss_batch_=get_nn_loss_batch_)
