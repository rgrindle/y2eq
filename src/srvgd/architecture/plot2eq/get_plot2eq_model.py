from srvgd.updated_eqlearner.tokenization_rg import token_map, token_map_2d
from srvgd.architecture.plot2eq.models import Encoder, Encoder_3d, DecoderWithAttention

import torch

import os


def get_plot2eq_model(model_name, path, device,
                      emb_dim=512,
                      attention_dim=512,
                      decoder_dim=512,
                      vocab_size=None,
                      dropout=0.25,
                      resnet_num=18,
                      two_d=False):

    if two_d:
        encoder = Encoder_3d(resnet_num)
        vocab_size = len(token_map_2d)
    else:
        encoder = Encoder(resnet_num)
        vocab_size = len(token_map)
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=vocab_size,
                                   encoder_dim=encoder.out_shape,
                                   dropout=dropout)

    checkpoint = torch.load(os.path.join(path, model_name),
                            map_location=device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    return encoder, decoder
