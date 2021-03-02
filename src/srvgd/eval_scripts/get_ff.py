"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 2, 2021

PURPOSE: Put y values into NN and get functional forms

NOTES: y-values come from saved location (see get_x_y)

TODO:
"""
from srvgd.utils.eval import translate_sentence
from srvgd.eval_scripts.beam_search import beam_search

import torch
import pandas as pd


def get_ff(y_int_list, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_data = []
    for i, y_int in enumerate(y_int_list):

        predicted = translate_sentence(sentence=y_int,
                                       model=model,
                                       device=device,
                                       max_len=67)[0]
        predicted_data.append(predicted[5:-3])
        # print('.', flush=True, end='')
        print(i, predicted_data[-1])

    pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)


def get_ff_beam_search(y_int_list, model, beam_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicted_data = []
    for i, y_int in enumerate(y_int_list):

        predicted = beam_search(beam_size=beam_size,
                                encoder_input=torch.Tensor(y_int),
                                model=model,
                                device=device)

        predicted_data.append(predicted[5:-3])

        # print('.', flush=True, end='')
        print(i, predicted_data[-1])

    pd.DataFrame(predicted_data).to_csv('01_predicted_ff_beam{}.csv'.format(beam_size), index=False, header=None)
