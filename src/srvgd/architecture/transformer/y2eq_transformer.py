"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 14, 2021

PURPOSE: Create a version of y2eq that is a
         transformer.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.transformer import Transformer
from srvgd.architecture.transformer.train_and_validate import get_dataset, split_dataset, train_many_epochs
from srvgd.updated_eqlearner.tokenization_rg import token_map

import torch
import torch.nn as nn


class y2eqTransformer(nn.Module):
    def __init__(self,
                 src_input_size=1,
                 src_max_len=30,
                 embedding_size=512,
                 trg_vocab_size=len(token_map),
                 num_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 forward_expansion=512,
                 dropout=0.1,
                 trg_max_len=100,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(y2eqTransformer, self).__init__()

        self.transformer = Transformer(src_input_size,
                                       src_max_len,
                                       embedding_size,
                                       trg_vocab_size,
                                       num_heads,
                                       num_encoder_layers,
                                       num_decoder_layers,
                                       forward_expansion,
                                       dropout,
                                       trg_max_len,
                                       device)

        self.src_word_embedding = nn.Linear(src_input_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

    def forward(self, src, trg):
        src_embedding = self.src_word_embedding(src)
        trg_embedding = self.trg_word_embedding(trg)
        return self.transformer(src_embedding, trg_embedding)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y2eq_trans_model = y2eqTransformer().to(device)

if __name__ == '__main__':
    # load_model = 'y2eq_transformer.pt'

    # Get number of trainable parameters
    num_params = sum(p.numel() for p in y2eq_trans_model.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    dataset_name = 'dataset_train_ff1000.pt'
    dataset = get_dataset(dataset_name, device)
    train_iterator, valid_iterator = split_dataset(dataset)

    # if load_model is not None:
    #     checkpoint = torch.load('../../../models/'+load_model, map_location=device)
    #     print(checkpoint.keys())
    #     exit()
    #     y2eq_trans_model.load_state_dict(checkpoint[''])

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=y2eq_trans_model,
                      device=device,
                      model_name='y2eq_transformer.pt')
