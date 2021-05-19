"""
AUTHOR: Ryan Grindle (modified from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer)

LAST MODIFIED: May 14, 2021

PURPOSE: A seq2seq transformers including positional
         encodings in the form of embedding layers.

NOTES: As is this is perfect for y2eq, but will need
       to be used or in conjuction with ResNet for
       plot2eq.

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import token_map

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""

    def __init__(self, vocab_size, embedding_size,
                 max_len, device):
        super().__init__()

        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device

    def forward(self, x):
        """Expected shape: x.shape = (batch size, seq length, embedding size)"""

        positions = (
            torch.arange(0, x.shape[1])
            .repeat(x.shape[0], 1)
            .to(self.device)
        )

        return x + self.position_embedding(positions)


class Transformer(nn.Module):
    def __init__(self,
                 src_input_size,
                 src_max_len,
                 embedding_size=512,
                 trg_vocab_size=len(token_map),
                 num_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 forward_expansion=512,
                 dropout=0.1,
                 trg_max_len=100,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super(Transformer, self).__init__()

        self.src_pos_enc = PositionalEncoding(vocab_size=src_input_size,
                                              embedding_size=embedding_size,
                                              max_len=src_max_len,
                                              device=device)
        self.trg_pos_enc = PositionalEncoding(vocab_size=trg_vocab_size,
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

    def forward(self, src, trg):

        embed_src = self.dropout(self.src_pos_enc(src))
        embed_trg = self.dropout(self.trg_pos_enc(trg))

        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[1]-1).to(self.device)
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


if __name__ == '__main__':
    model = Transformer(src_input_size=1,
                        src_max_len=30)

    model(torch.zeros((1, 30, 1)), torch.zeros((1, len(token_map), 1)))
