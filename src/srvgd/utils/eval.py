"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 15, 2021

PURPOSE: Evaluate the model after training.

NOTES:

TODO:
"""
from eqlearner.dataset.processing.tokenization import default_map, reverse_map
from srvgd.architecture.seq2seq_cnn_attention import MAX_OUTPUT_LENGTH
from srvgd.common.save_load_dataset import load_and_format_dataset, onehot2token

from scipy.optimize import minimize
import cma
import re
from tensorflow import keras
import numpy as np
import sympy
from sympy import sin, log, exp, cos, cosh, E  # noqa: F401

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


def get_string(string, mapping=None, sym_mapping=None):
    """Modified version of
    eqlearner.dataset.processing.tokenization.get_string.
    This version places 'END' and removes everything after it."""
    if not mapping:
        tmp = default_map()
        mapping = reverse_map(tmp, symbols=sym_mapping)
    mapping_string = mapping.copy()
    mapping_string[12] = 'START'
    mapping_string[13] = 'END'
    curr = ''.join([mapping_string[digit] for digit in string])
    if len(string) < 2:
        return RuntimeError
    if len(string) == 2:
        return 0
    end = curr.find('END')
    return curr[:end]


def decode(output):
    onehots = get_onehot_from_softmax(output)

    eq_str_list = []
    for obs in onehots:
        eq_num_list = [onehot2token[tuple(onehot)] for onehot in obs]
        eq_str = get_string([t for t in eq_num_list if t != 0])
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


def is_valid_eq(eq_list, support):
    mask = []
    for eq in eq_list:
        try:
            f = get_f(eq)
            f(support)
            mask.append(True)
        except (SyntaxError, TypeError, AttributeError):
            mask.append(False)
        print(eq, mask[-1])
    return np.array(mask)


def eval_model(model, inputs, support):
    output = model.predict(inputs)
    decoded_output = decode(output)
    mask = is_valid_eq(decoded_output, support)
    num_invalid = len(mask)-sum(mask)
    print('Found {} invalid equations'.format(num_invalid))
    return decoded_output, mask


def load_model(model_name):
    return keras.models.load_model(os.path.join('..', '..', '..', 'models', 'model_'+model_name))
    # trained_model = keras.models.load_model(os.path.join('models', 'model_'+model_name))
    # test_model.set_weights(trained_model.get_weights())
    # return test_model


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
    return c_eq, coeff_index


def RMSE(y, y_hat):
    return np.sqrt(np.mean(np.power(y-y_hat, 2)))


# def regression(f_hat, y, num_coeffs, support):
#     def loss(c, x):
#         y_hat = f_hat(c, x)
#         return RMSE(normalize(y_hat), y)

#     bestever = cma.optimization_tools.BestSolution()
#     for popsize in [100]:
#         es = cma.CMAEvolutionStrategy(np.ones(num_coeffs),
#                                       0.5,
#                                       {'popsize': popsize,
#                                        'verb_append': bestever.evalsall})

#         while not es.stop():
#             solutions = es.ask()
#             es.tell(solutions, [loss(c=s, x=support) for s in solutions])
#             es.disp()

#         bestever.update(es.best)

#         if bestever.f < 1e-8:  # global optimum was hit
#             break

#     return bestever.x, bestever.f


def regression(f_hat, y, num_coeffs, support):
    def loss(c, x):
        y_hat = f_hat(c, x)
        return RMSE(normalize(y_hat), y)

    res = minimize(loss, np.ones(num_coeffs), args=(support,), method='BFGS')
    return res.x, res.fun


def normalize(y):
    min_ = np.min(y)
    return (y-min_)/(np.max(y)-min_)


def fit_eq(eq_list, support, y_list):
    x = sympy.symbols('x')  # noqa: F841
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
        f_list.append(lambda x: normalize(f_hat(x=x, c=coeffs)))

    return coeff_list, rmse_list, f_list


if __name__ == '__main__':
    import pandas as pd

    x, _, info = load_and_format_dataset('dataset_no_scaling', dataset_type='test', return_info=True)
    pad(x)  # depending on dataset max decoder lenght may vary
    model = load_model('seq2seq_cnn_attention_model_dataset_no_scaling')
    decoded_output, mask = eval_model(model,
                                      inputs=x,
                                      support=np.array(info['Support']))

    # scaler = MinMaxScaler()
    # scaler.min_ = info['min_']
    # scaler.scale_ = info['scale_']
    # unscaled_x = scaler.inverse_transform(x[0][:, :, 0])
    unscaled_x = x[0]

    print(decoded_output[mask] == sum(mask))
    _, rmse_list, _ = fit_eq(decoded_output[mask],
                             np.array(info['Support']),
                             unscaled_x[mask])
    pd.DataFrame(rmse_list).to_csv('rmse_test.csv')
