"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 24, 2021

PURPOSE: Create eq2y as a transformer

NOTES:

TODO:
"""
from srvgd.architecture.transformer.transformer import PositionalEncoding
from srvgd.architecture.transformer.train_and_validate import get_dataset, split_dataset, train_many_epochs
from srvgd.updated_eqlearner.tokenization_rg import token_map

import torch
import torch.nn as nn


class eq2yTransformer(nn.Module):
    def __init__(self,
                 src_input_size=len(token_map),
                 src_max_len=100,
                 embedding_size=512,
                 x_vocab_size=1,
                 trg_vocab_size=1,
                 num_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 forward_expansion=512,
                 dropout=0.1,
                 x_max_len=30,
                 trg_max_len=30,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super(eq2yTransformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_input_size, embedding_size)
        self.x_word_embedding = nn.Linear(x_vocab_size, embedding_size)

        self.src_pos_enc = PositionalEncoding(vocab_size=src_input_size,
                                              embedding_size=embedding_size,
                                              max_len=src_max_len,
                                              device=device)
        self.x_pos_enc = PositionalEncoding(vocab_size=trg_vocab_size,
                                            embedding_size=embedding_size,
                                            max_len=trg_max_len,
                                            device=device)

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

    def forward(self, eq):
        # x is fixed in here and used like target
        x = torch.arange(0.1, 3.1, 0.1).view(1, 30, 1)
        x = torch.repeat_interleave(x, len(eq), axis=0).to(self.device)
        src_embedding = self.src_word_embedding(eq)
        x_embedding = self.x_word_embedding(x)

        embed_src = self.dropout(self.src_pos_enc(src_embedding))
        embed_x = self.dropout(self.x_pos_enc(x_embedding))
        # embed_src.shape = [batch size, src seq length, feature number]
        # embed_x.shape = [batch size, x seq length, feature number]

        embed_src = embed_src.permute(1, 0, 2)
        embed_x = embed_x.permute(1, 0, 2)

        # model expects shape = [sequence length, batch size, feature number]
        # so, thus the permute above

        out = self.transformer(
            embed_src,
            embed_x,
            tgt_mask=None,
        )
        out = self.fc_out(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eq2y_trans_model = eq2yTransformer().to(device)


def get_nn_loss_batch_eq2y(batch, model, device, criterion):
    target = batch[0].to(device)
    inp_data = batch[1].to(device)

    # Forward prop
    output = model(inp_data).permute(1, 0, 2)
    loss = criterion(output, target)
    return loss


if __name__ == '__main__':
    checkpoint_filename = 'eq2y_transformer_more_ff_2000.pt'
    # checkpoint_filename = None

    # Get number of trainable parameters
    num_params = sum(p.numel() for p in eq2y_trans_model.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    dataset_name = 'dataset_train_ff1000_no_coeffs2.pt'
    dataset = get_dataset(dataset_name, device)

    train_iterator, valid_iterator = split_dataset(dataset,
                                                   split=(5287, 2266))

    model_name = 'eq2y_transformer_more_ff_2000.pt'
    kwargs = {}
    if checkpoint_filename is not None:
        checkpoint = torch.load('../../../../models/'+checkpoint_filename,
                                map_location=device)
        eq2y_trans_model.load_state_dict(checkpoint['state_dict'])

        model_name = 'eq2y_transformer_more_ff_4000.pt'
        kwargs = {'train_losses': checkpoint['train_loss'],
                  'valid_losses': checkpoint['val_loss'],
                  'optimizer_state_dict': checkpoint['optimizer']}

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=eq2y_trans_model,
                      device=device,
                      model_name=model_name,
                      criterion=torch.nn.MSELoss(),
                      num_epochs=2000,
                      get_nn_loss_batch_=get_nn_loss_batch_eq2y,
                      **kwargs)
