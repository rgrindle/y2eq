"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 6, 2021

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

    # if isinstance(sentence, str):
    #     numerized_tokens = tokenize_eq(sentence)
    #     # nlp = spacy.load('de')
    #     # tokens = [token.text.lower() for token in nlp(sentence)]
    # else:
    #     # tokens = [token.lower() for token in sentence]
    #     numerized_tokens = sentence

    # Apply start and end tokens
    # tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # # Convert to integer representation of tokens
    # src_indexes = [src_field.vocab.stoi[token] for token in tokens]

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


file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
# file_endname = '_epochs100_0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, 'cnn{}.pt'.format(file_endname))
model.eval()

test_data = torch.load('dataset_test.pt', map_location=device)
# test_data = torch.load('test_data_int_comp.pt', map_location=device)

predicted_data = []
for obs in test_data:
    inputs, targets = obs
    predicted = translate_sentence(sentence=inputs,
                                   model=model,
                                   device=device)[0]
    predicted_data.append(predicted)
    print('.', flush=True, end='')

pd.DataFrame(predicted_data).to_csv('test_output{}.csv'.format(file_endname), index=False)
