"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 13, 2021

PURPOSE: Evaluate the model after training.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string, token_map, token_map_2d
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.data_gathering.get_normal_x import get_normal_x
from srvgd.architecture.torch.get_model import get_model
from srvgd.utils.rmse import RMSE

import torch
import numpy as np
import pandas as pd

import json


def write_x_y_lists(eq_list_filename, x_type):
    assert x_type in ('fixed', 'different', 'normal')

    np.random.seed(0)

    eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

    if x_type == 'fixed':
        x_int = np.arange(0.1, 3.1, 0.1)

    x_ext = np.arange(3.1, 6.1, 0.1)

    x_list = []
    y_int_normalized_list = []
    y_int_unnormalized_list = []
    y_ext_unnormalized_list = []
    for i, eq in enumerate(eq_list):

        if x_type == 'different':
            x_int = np.random.uniform(0.1, 3.1, 30)
        elif x_type == 'normal':
            x_int = get_normal_x(num_points=30)

        x_list.append(x_int.tolist())

        eq = EquationInfix(eq, apply_coeffs=False)
        y_int = eq.f(x_int)

        if np.any(np.isnan(y_int)):
            print('found nan')
            exit()

        y_int_unnormalized_list.append(y_int.tolist())
        y_int_normalized_list.append(normalize(y_int)[:, None].tolist())
        y_ext_unnormalized_list.append(eq.f(x_ext).tolist())

    print(len(x_list))
    assert len(x_list) == len(y_int_normalized_list)
    assert len(x_list) == len(y_int_unnormalized_list)
    assert len(x_list) == len(y_ext_unnormalized_list)

    with open('00_x_list.json', 'w') as file:
        json.dump(x_list, file)

    with open('00_y_int_normalized_list.json', 'w') as file:
        json.dump(y_int_normalized_list, file)

    with open('00_y_int_unnormalized_list.json', 'w') as file:
        json.dump(y_int_unnormalized_list, file)

    with open('00_y_ext_unnormalized_list.json', 'w') as file:
        json.dump(y_ext_unnormalized_list, file)


def eval_nn(input_list, model_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device,
                      path='../models/',
                      load_weights=model_filename)

    predicted_data = []
    for i, input_ in enumerate(input_list):

        predicted = translate_sentence(sentence=input_,
                                       model=model,
                                       device=device,
                                       max_len=67)[0]
        predicted_data.append(predicted[5:-3])
        print(i, predicted_data[-1])

    pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)


def fit_coeffs_and_get_rmse(y_int_list, y_ext_list, ff_list):
    x_int = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    rmse_int_list = []
    rmse_ext_list = []
    for i, (ff, y_int, y_ext) in enumerate(zip(ff_list, y_int_list, y_ext_list)):
        y_int = np.array(y_int).flatten()

        try:
            eq = EquationInfix(ff, x=x_int)
            eq.fit(y_int)

            y_int_true_norm, true_min_, true_scale = normalize(y_int, return_params=True)
            y_int_pred_norm = normalize(eq.f(c=eq.coeffs, x=x_int).flatten(), true_min_, true_scale)

            y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
            y_ext_pred_norm = normalize(eq.f(c=eq.coeffs, x=x_ext).flatten(), true_min_, true_scale)

            rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)
            rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)

        except SyntaxError:
            rmse_int = np.inf
            rmse_ext = np.inf

        rmse_int_list.append(rmse_int)
        rmse_ext_list.append(rmse_ext)
        print(i, rmse_int_list[-1], rmse_ext_list[-1])

    pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('02_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])


def translate_sentence(sentence, model, device, max_len=100,
                       two_d=False):

    model.eval()

    if type(sentence) != torch.Tensor:
        src_tensor = torch.Tensor(sentence).to(device)
    else:
        src_tensor = sentence.to(device)

    src_tensor = src_tensor.unsqueeze(0)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    if two_d:
        mapping = token_map_2d
    else:
        mapping = token_map
    trg_indexes = [mapping['START']]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == mapping['END']:
            break

    trg_tokens = get_eq_string(trg_indexes, two_d)

    return trg_tokens, attention


if __name__ == '__main__':
    pass
