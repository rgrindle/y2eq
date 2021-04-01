"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 15, 2021

PURPOSE: Evaluate the model after training.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string, token_map, token_map_2d
from srvgd.architecture.seq2seq_cnn_attention import MAX_OUTPUT_LENGTH
from srvgd.common.save_load_dataset import load_and_format_dataset, onehot2token

import torch
from scipy.optimize import minimize
import re
from tensorflow import keras
import numpy as np
# from sympy import sin, log, exp, cos, cosh, E  # noqa: F401

import os


def get_onehot_from_num(num, length):
    onehot = np.zeros(length)
    onehot[num] = 1
    return onehot


def get_onehot_from_softmax(softmax):
    onehot_locs = np.argmax(softmax, axis=2)
    onehots = []
    length = softmax.shape[2]
    for obs in onehot_locs:
        onehot_eq = []
        for token in obs:
            onehot_eq.append(get_onehot_from_num(token, length))
        onehots.append(onehot_eq)
    return np.array(onehots)


def decode(output):
    onehots = get_onehot_from_softmax(output)

    eq_str_list = []
    for obs in onehots:
        eq_num_list = [onehot2token[tuple(onehot)] for onehot in obs]
        eq_str = get_eq_string([t for t in eq_num_list if t != 0])
        eq_str_list.append(eq_str)
    return np.array(eq_str_list)


def get_f(eq):
    for prim in ['sin', 'log', 'exp']:
        eq = eq.replace(prim, 'np.'+prim)
    eq = eq.replace('E', 'np.e')
    if 'c[' in eq:
        lambda_str_beg = 'lambda c, x:'
    else:
        lambda_str_beg = 'lambda x:'
    f = eval(lambda_str_beg+eq)
    return f


def is_valid_eq_list(eq_list, support):
    mask = []
    for eq in eq_list:
        try:
            f = get_f(eq)
            f(support)
            mask.append(True)
        except (SyntaxError, TypeError, AttributeError):
            mask.append(False)
    return np.array(mask)


def eval_model(model, inputs, support):
    output = model.predict(inputs)
    decoded_output = decode(output)
    mask = is_valid_eq_list(decoded_output, support)
    num_invalid = len(mask)-sum(mask)
    print('Found {} invalid equations'.format(num_invalid))
    return decoded_output, mask


def load_model(model_name):
    return keras.models.load_model(os.path.join('..', '..', '..', 'models', 'model_'+model_name))


def pad(x):
    x[1] = np.hstack((x[1], np.zeros((x[1].shape[0], MAX_OUTPUT_LENGTH-x[1].shape[1], x[1].shape[2]))))


def apply_coeffs(eq):
    """Take eq (str without coeffs although possibly
    hard-coded ones) and put c[0], c[1], ... where
    applicable.

    Returns
    -------
    eq_c : str
        equation str including adjustable coefficients.
    num_coeff : int
        The number of coefficients placed.

    Examples
    --------
    >>> apply_coeffs('x')
    ('c[0]*x', 1)

    >>> apply_coeffs('sin(x)')
    ('c[1]*sin(c[0]*x)', 2)

    >>> apply_coeffs('sin(exp(x))')
    ('c[1]*sin(c[2]*exp(c[0]*x))', 3)
    """
    coeff_index = 0

    # First, attach a coefficient to every occurance of x.
    # Be careful to find the variable not the x in exp, for example.
    eq_str_list = []
    for i, e in enumerate(eq):
        if e == 'x' and (i == 0 or eq[i-1] != 'e'):
            eq_str_list.append('c[{}]*x'.format(coeff_index))
            coeff_index += 1
        else:
            eq_str_list.append(e)

    # Put a coefficient in front of every term.
    c_eq = ''.join(eq_str_list)
    c_eq_str_list = []
    for term in c_eq.split('+'):
        if 'c[' == term[0:2]:
            c_eq_str_list.append(term)
        else:
            c_eq_str_list.append('c[{}]*'.format(coeff_index)+term)
            coeff_index += 1
    c_eq = '+'.join(c_eq_str_list)

    # Put a coeff in front of any missed primitives.
    # Without this block sin(sin(x)) -> c[1]*sin(sin(c[0]*x))
    # but with this block sin(sin(x)) -> c[1]*sin(c[2]*sin(c[0]*x))
    for prim in ['sin', 'exp', 'log']:
        c_eq_str_list = []
        prev_i = 0
        for m in re.finditer(prim, c_eq):
            i = m.start()
            c_eq_str_list.append(c_eq[prev_i:i])
            if c_eq[i-2:i] != ']*':
                c_eq_str_list.append('c[{}]*'.format(coeff_index))
                coeff_index += 1
            prev_i = i
        c_eq_str_list.append(c_eq[prev_i:])
        c_eq = ''.join(c_eq_str_list)

    # Add verticle shift
    # c_eq += '+c[{}]'.format(coeff_index)
    return c_eq, coeff_index


def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.power(y-y_hat, 2)))


def regression(f_hat, y, num_coeffs, support):
    def loss(c, x):
        y_hat = f_hat(c, x).flatten()
        return RMSE(y_hat, y)

    res = minimize(loss, np.ones(num_coeffs), args=(support,), bounds=[(-3, 3)]*num_coeffs,
                   method='L-BFGS-B')
    return res.x, loss(res.x, support)


def fit_eq(eq_list, support, y_list):
    # x = sympy.symbols('x')  # noqa: F841
    coeff_list = []
    rmse_list = []
    f_list = []
    for eq, y in zip(eq_list, y_list):
        print('eq', eq)
        eq_c, num_coeffs = apply_coeffs(eq)
        print('eq_c', eq_c)
        print()
        f_hat = get_f(eq_c)
        coeffs, rmse = regression(f_hat, y, num_coeffs, support)
        coeff_list.append(coeffs)
        rmse_list.append(rmse)

        f_str = eq_c
        for i in range(num_coeffs):
            f_str = f_str.replace('c[{}]'.format(i), str(coeffs[i]))
        f_hat_fixed_coeffs = get_f(f_str)
        f_list.append(f_hat_fixed_coeffs)

    return coeff_list, rmse_list, f_list


def is_eq_valid(eq_str, x=np.arange(0.1, 3.1, 0.1)):
    try:
        f = get_f(eq_str)
        y_hat_values = f(x)
        return type(y_hat_values) != np.ufunc
    except (SyntaxError, TypeError, AttributeError, NameError, FloatingPointError, ValueError):
        return False


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
    import pandas as pd

    x, _, info = load_and_format_dataset('dataset_no_scaling', dataset_type='test', return_info=True)
    pad(x)  # depending on dataset max decoder lenght may vary
    model = load_model('seq2seq_cnn_attention_model_dataset_no_scaling')
    decoded_output, mask = eval_model(model,
                                      inputs=x,
                                      support=np.array(info['Support']))

    unscaled_x = x[0]

    print(decoded_output[mask] == sum(mask))
    _, rmse_list, _ = fit_eq(decoded_output[mask],
                             np.array(info['Support']),
                             unscaled_x[mask])
    pd.DataFrame(rmse_list).to_csv('rmse_test.csv')
