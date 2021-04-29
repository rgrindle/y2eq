"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 13, 2021

PURPOSE: Evaluate the model after training.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string, token_map, token_map_2d, token_map_with_coeffs
from equation.EquationInfix import EquationInfix
from srvgd.utils.normalize import normalize
from srvgd.data_gathering.get_normal_x import get_normal_x
from srvgd.architecture.y2eq.get_y2eq_model import get_y2eq_model
from srvgd.architecture.plot2eq.get_plot2eq_model import get_plot2eq_model
from srvgd.utils.rmse import RMSE

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

import json


def write_x_y_lists(eq_list_filename, x_type):
    assert x_type in ('fixed', 'different', 'normal')

    np.random.seed(0)

    eq_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()

    if x_type == 'fixed':
        x_int = np.arange(0.1, 3.1, 0.1)

    x_int_fixed = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    x_list = []
    y_int_normalized_list = []
    y_int_unnormalized_list = []
    y_int_fixed_unnormalized_list = []
    y_ext_unnormalized_list = []
    for i, eq in enumerate(eq_list):

        if x_type == 'different':
            x_int = np.random.uniform(0.1, 3.1, 30)
            x_int.sort()
        elif x_type == 'normal':
            x_int = get_normal_x(num_points=30)
            x_int.sort()

        x_list.append(x_int.tolist())

        eq = EquationInfix(eq, apply_coeffs=False)
        y_int = eq.f(x_int)

        if np.any(np.isnan(y_int)):
            print('found nan')
            exit()

        y_int_unnormalized_list.append(y_int.tolist())
        y_int_fixed_unnormalized_list.append(eq.f(x_int_fixed).tolist())
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

    with open('00_y_int_fixed_unnormalized_list.json', 'w') as file:
        json.dump(y_int_fixed_unnormalized_list, file)

    with open('00_y_ext_unnormalized_list.json', 'w') as file:
        json.dump(y_ext_unnormalized_list, file)


def eval_y2eq(input_list, model_filename, **get_model_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_y2eq_model(device,
                           path='../models/',
                           load_weights=model_filename,
                           **get_model_kwargs)

    two_d = get_model_kwargs['two_d'] if 'two_d' in get_model_kwargs else False
    include_coeffs = get_model_kwargs['include_coeffs'] if 'include_coeffs' in get_model_kwargs else False

    predicted_data = []
    for i, input_ in enumerate(input_list):

        predicted = translate_sentence(sentence=input_,
                                       model=model,
                                       device=device,
                                       two_d=two_d,
                                       include_coeffs=include_coeffs)[0]
        predicted_data.append(predicted[5:-3])
        print(i, predicted_data[-1])

    pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)


def eval_plot2eq(input_list, model_filename, **get_model_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = get_plot2eq_model(device=device,
                                         path='../models/',
                                         model_name=model_filename,
                                         **get_model_kwargs)

    two_d = False if 'two_d' not in get_model_kwargs else get_model_kwargs['two_d']

    predicted_data = []
    for i, input_ in enumerate(input_list):

        predicted = evaluate_one_image(input_, encoder, decoder, device, two_d)
        print(predicted)
        predicted_data.append(predicted)
        print(i, predicted_data[-1])

    pd.DataFrame(predicted_data).to_csv('01_predicted_ff.csv', index=False, header=None)


def fit_coeffs_and_get_rmse(y_int_fixed_list, y_ext_list, ff_list):
    x_int = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    rmse_int_list = []
    rmse_ext_list = []
    for i, (ff, y_int_fixed, y_ext) in enumerate(zip(ff_list, y_int_fixed_list, y_ext_list)):
        y_int = np.array(y_int_fixed).flatten()

        if pd.isnull(ff):
            rmse_int = np.inf
            rmse_ext = np.inf

        else:
            eq = EquationInfix(ff, x=x_int)

            if eq.is_valid():
                eq.fit(y_int)

                y_int_true_norm, true_min_, true_scale = normalize(y_int, return_params=True)
                y_int_pred_norm = normalize(eq.f(c=eq.coeffs, x=x_int).flatten(), true_min_, true_scale)

                y_ext_true_norm = normalize(y_ext, true_min_, true_scale)
                y_ext_pred_norm = normalize(eq.f(c=eq.coeffs, x=x_ext).flatten(), true_min_, true_scale)

                rmse_int = RMSE(y_int_true_norm, y_int_pred_norm)
                rmse_ext = RMSE(y_ext_true_norm, y_ext_pred_norm)

            else:
                rmse_int = np.inf
                rmse_ext = np.inf

        rmse_int_list.append(rmse_int)
        rmse_ext_list.append(rmse_ext)
        print(i, rmse_int_list[-1], rmse_ext_list[-1])

    pd.DataFrame([rmse_int_list, rmse_ext_list]).T.to_csv('02_rmse.csv', index=False, header=['rmse_int', 'rmse_ext'])


def translate_sentence(sentence, model, device, max_len=100,
                       two_d=False, include_coeffs=False):
    assert not (two_d and include_coeffs)

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
    elif include_coeffs:
        mapping = token_map_with_coeffs
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

    trg_tokens = get_eq_string(trg_indexes, two_d, include_coeffs)

    return trg_tokens, attention


def evaluate_one_image(image, encoder, decoder, device, two_d):

    if type(image) != torch.Tensor:
        image = torch.Tensor(image)

    image = image.to(device).unsqueeze(0)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # flatten encoding
    encoder_dim = encoder_out.size(-1)
    encoder_out = encoder_out.contiguous().view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    if two_d:
        mapping = token_map_2d
    else:
        mapping = token_map

    prev_word = torch.LongTensor([mapping['START']]).to(device)
    sentence = []

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        prev_word, h, c = decode_one_token(decoder, prev_word, h, c, encoder_out)
        sentence.append(prev_word.item())

        if step > 100 or prev_word == mapping['END']:
            break
        step += 1

    return get_eq_string(sentence, two_d).replace('END', '')


def decode_one_token(decoder, prev_word, h, c, encoder_out):

    embeddings = decoder.embedding(prev_word.unsqueeze(0)).squeeze(1)  # (s, embed_dim)

    awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

    gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
    awe = gate * awe

    h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

    pred_raw = decoder.fc(h)  # (s, vocab_size)
    pred_raw = F.log_softmax(pred_raw, dim=1)
    prev_word = torch.argmax(pred_raw).unsqueeze(0)
    return prev_word, h, c


def remove_nan(data):
    return data[~np.isnan(data)]


if __name__ == '__main__':
    pass
