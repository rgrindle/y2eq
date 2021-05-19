"""
AUTHOR: Ryan Grindle

LAST MODIFIED: May 14, 2021

PURPOSE: Implement a transformer version of plot2eq.
         This implementation will use a ResNet before
         sending data into the transformer.

NOTES:

TODO:
"""
from srvgd.architecture.transformer.transformer import Transformer
from srvgd.architecture.transformer.train_and_validate import get_dataset, split_dataset, train_many_epochs
from srvgd.updated_eqlearner.tokenization_rg import token_map
from srvgd.data_gathering.change_input_to_image import change_input_to_image

import torch
import torchvision
import torch.nn as nn


class plot2eqTransformer(nn.Module):
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
        super(plot2eqTransformer, self).__init__()

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

        # Get ResNet, but remove linear and
        # pool layers (since we're not doing
        # classification)
        resnet = torchvision.models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.src_word_embedding = nn.Linear(src_input_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

    def forward(self, src, trg):
        src_embedding = self.resnet(src)
        # src_embedding.shape = (batch size, 512, reduced img width, reduced img height)

        src_embedding = src_embedding.permute(0, 2, 3, 1)
        # src_embedding.shape = (batch size, reduced img width, reduced img height, 512)

        batch_size, _, _, embedding_size = src_embedding.shape
        src_embedding = src_embedding.view(batch_size, -1, embedding_size)
        # src_embedding.shape = (batch size, (reduced img width) * (reduced img height), 512)

        trg_embedding = self.trg_word_embedding(trg)
        return self.transformer(src_embedding, trg_embedding)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plot2eq_trans_model = plot2eqTransformer().to(device)

if __name__ == '__main__':
    # Get number of trainable parameters
    num_params = sum(p.numel() for p in plot2eq_trans_model.parameters() if p.requires_grad)
    print('Num trainable params:', num_params)

    # Get dataset
    dataset_name = 'dataset_train_ff1000.pt'
    dataset = get_dataset(dataset_name, device)
    dataset = change_input_to_image(dataset, image_size=(64, 64))
    print(len(dataset))
    train_iterator, valid_iterator = split_dataset(dataset)

    # if load_model:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), plot2eq_trans_model, optimizer)

    train_many_epochs(train_iterator=train_iterator,
                      valid_iterator=valid_iterator,
                      model=plot2eq_trans_model,
                      device=device,
                      model_name='plot2eq_transformer.pt')
