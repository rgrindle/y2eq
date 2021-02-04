"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Feb 4, 2021

PURPOSE: Given a dataset train numeric NN and get gridified
         y-values to be input to the symbolic NN.

NOTES: gridified means x = [0.1, 0.2, ..., 3.0]

TODO:
"""
from srvgd.utils.eval import translate_sentence, get_f, normalize, RMSE
from srvgd.architecture.torch.get_model import get_model
from equation.EquationInfix import EquationInfix

import torch
from tensorflow import keras
import numpy as np
import pandas as pd

import os


def gridify(model, x_grid):
    return model.predict(x_grid)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int,
                        help='index in dataset to choose which equation to do.')
    args = parser.parse_args()

    x_int = np.arange(0.1, 3.1, 0.1)
    x_ext = np.arange(3.1, 6.1, 0.1)

    index = 0
    model_name = 'function_approximation_index{}'.format(args.index)
    numeric_model = keras.models.load_model(os.path.join('..', 'numeric_regression', 'models', 'model_'+model_name))
    y_input = gridify(numeric_model, x_int)

    file_endname = '_dataset_train_ff1000_batchsize2000_lr0.0001_clip1_layers10_900'
    device = torch.device('cpu')
    symbolic_model = get_model(device,
                               path=os.path.join('..', 'models'),
                               load_weights='cnn{}.pt'.format(file_endname))
    predicted_eq = translate_sentence(sentence=torch.Tensor(y_input),
                                      model=symbolic_model,
                                      device=device)[0][5:-3]
    print('predicted functional form', predicted_eq)

    eq = EquationInfix(predicted_eq)

    eq_true_list = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).values.flatten()
    eq_true = eq_true_list[index]
    print('true equation', eq_true)
    f_true = get_f(eq_true)
    y_int_true = f_true(x_int)

    eq.fit(y_int_true)
    print('interp rmse', eq.rmse)

    _, true_min_, true_scale = normalize(y_int_true, return_params=True)

    y_int_true_normalized = normalize(y_int_true, true_min_, true_scale)

    y_ext_true = f_true(x_ext)
    y_ext_true_normalized = normalize(y_ext_true, true_min_, true_scale)

    y_int_pred = eq.f(eq.coeffs, x_int)
    y_int_pred_normalized = normalize(y_int_pred, true_min_, true_scale)

    y_ext_pred = eq.f(eq.coeffs, x_ext)
    y_ext_pred_normalized = normalize(y_ext_pred, true_min_, true_scale)

    int_rmse = RMSE(y_int_true, y_int_pred)
    int_normalized_rmse = RMSE(y_int_true_normalized, y_int_pred_normalized)
    ext_rmse = RMSE(y_ext_true, y_ext_pred)
    ext_normalized_rmse = RMSE(y_ext_true_normalized, y_ext_pred_normalized)

    print('int_rmse', int_rmse)
    print('int_normalized_rmse', int_normalized_rmse)
    print('ext_rmse', ext_rmse)
    print('ext_normalized_rmse', ext_normalized_rmse)

    os.makedirs('rmse', exist_ok=True)

    with open('rmse/{}.txt'.format(args.index), 'w') as f:
        f.write('int_normalized_rmse,int_rmse,ext_normalized_rmse,ext_rmse\n')
        f.write('{},{},{},{}'.format(int_normalized_rmse, int_rmse, ext_normalized_rmse, ext_rmse))

    # import matplotlib.pyplot as plt
    # plt.plot(x_int, y_input, '.-')
    # plt.plot(x_int, true_y, '.-')
    # plt.plot(x_int, eq.f(eq.coeffs, x_int), '.-')
    # plt.show()
