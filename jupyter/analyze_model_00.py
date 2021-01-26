"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 26, 2021

PURPOSE: Get output of trained NN on test dataset.

NOTES:

TODO:
"""
from eqlearner.dataset.processing.tokenization import get_string
from tokenization_rg import default_map
from get_model import get_model
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import itertools


def translate_sentence(sentence, model, device, max_len=100):

    model.eval()

    # src_tensor = torch.LongTensor(numerized_tokens).unsqueeze(0).to(device)
    src_tensor = sentence.unsqueeze(0)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    mapping = default_map()
    trg_indexes = [mapping['START']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == mapping['END']:
            break

    trg_tokens = get_string(trg_indexes)

    return trg_tokens, attention


if __name__ == '__main__':

    file_endname = '_dataset_train_ff_batchsize128_lr0.001_clip0.1_layers10_10'
    # file_endname = '_epochs100_0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device,
                      load_weights='cnn{}.pt'.format(file_endname))
    model.eval()

    test_data = torch.load('dataset_test_ff.pt', map_location=device)
    # test_data = torch.load('test_data_int_comp.pt', map_location=device)

    predicted_data = []
    for i, obs in enumerate(test_data):
        if i >= 1000:
            break

        inputs, targets = obs
        predicted = translate_sentence(sentence=inputs,
                                       model=model,
                                       device=device)[0]
        predicted_data.append(predicted)
        print('.', flush=True, end='')

    pd.DataFrame(predicted_data).to_csv('test_output{}.csv'.format(file_endname), index=False)
