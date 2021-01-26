"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 26, 2021

PURPOSE: Get output of trained NN on test dataset.

NOTES:

TODO:
"""
from srvgd.utils.eval import translate_sentence
from srvgd.architecture.torch.get_model import get_model
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import pandas as pd


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
