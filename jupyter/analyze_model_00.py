"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 28, 2021

PURPOSE: Get output of trained NN on test dataset.

NOTES:

TODO:
"""
from srvgd.utils.eval import translate_sentence
from srvgd.architecture.torch.get_model import get_model

import torch
import pandas as pd

import os


if __name__ == '__main__':

    file_endname = '_layers10_clip1_dropoutTrue_lr1e-4_no_duplicates_660'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device,
                      path=os.path.join('..', 'models'),
                      load_weights='cnn{}.pt'.format(file_endname))
    model.eval()

    test_data = torch.load(os.path.join('..', 'datasets', 'dataset_test_ff.pt'), map_location=device)
    # from eqlearner.dataset.processing.tokenization import get_string
    # import numpy as np
    # ff_list = [get_string(d[1].tolist())[5:-3] for d in test_data]
    # print(len(np.unique(ff_list)))
    # exit()
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
