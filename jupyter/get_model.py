from encoder import Encoder
from decoder import Decoder
from cnn_seq2seq_arch import Seq2Seq

import torch

INPUT_DIM = 1
OUTPUT_DIM = 22  # dictionary size
EMB_DIM = 256
HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10  # number of conv. blocks in encoder 10 original
DEC_LAYERS = 10  # number of conv. blocks in decoder 10 original
ENC_KERNEL_SIZE = 3  # must be odd!
DEC_KERNEL_SIZE = 3  # can be even or odd
ENC_DROPOUT = 0#0.25
DEC_DROPOUT = 0#0.25
TRG_PAD_IDX = 0
DEC_MAX_LENGTH = 67  # length of longest equation in terms of numbe of tokens


def get_model(device, load_weights=None):
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device, max_length=DEC_MAX_LENGTH)

    model = Seq2Seq(enc, dec).to(device)

    if load_weights is not None:
        model.load_state_dict(torch.load(load_weights, map_location=device))

    return model
