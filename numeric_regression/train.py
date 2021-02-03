"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 16, 2020

PURPOSE: Train a NN to do function approximation
         for a single function.

NOTES:

TODO: Separate plotting from training.
"""
from architecture_1d import model

from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow import keras

import os


def train(model, dataset, batch_size, epochs, index):
    model_name = 'function_approximation_index{}'.format(index)
    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('models', 'weights_'+model_name),
                                 monitor='val_loss')

    model.compile(optimizer='Adam', loss='mse')
    model.fit(dataset[0], dataset[1],
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True,
              callbacks=[model_cb, weights_cb])

    return keras.models.load_model(os.path.join('models', 'model_'+model_name))


def make_dataset(f, x_train, x_test):
    y_train, min_data, scale_ = normalize(f(x_train))
    train_dataset = [x_train[:, None], y_train[:, None]]
    y_test = normalize(f(x_test), min_data, scale_)[0]
    test_dataset = [x_test[:, None], y_test[:, None]]
    return train_dataset, test_dataset, min_data, scale_


def normalize(data, min_data=None, scale_=None):
    if min_data is None or scale_ is None:
        min_data = np.min(data)
        max_data = np.max(data)
        scale_ = 1./(max_data-min_data)
    return (data-min_data)*scale_, min_data, scale_


def unnormalize(data, min_data, scale_):
    return data/scale_+min_data


if __name__ == '__main__':
    import pandas as pd
    # import matplotlib.pyplot as plt

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int,
                        help='index in dataset to choose which equation to do.')
    args = parser.parse_args()

    eqs = pd.read_csv('../datasets/equations_with_coeff_test_ff1000.csv', header=None).iloc[:, 0].values

    batch_size = 1

    # make a dataset
    # np.random.seed(0)
    x_train = np.arange(0.1, 3.1, 0.1)
    x_test = np.arange(3.1, 6.1, 0.1)
    # f = lambda x: 3*x*np.sin(5*x)+7
    eq = eqs[args.index].replace('sin', 'np.sin').replace('log', 'np.log').replace('exp', 'np.exp')
    f = eval('lambda x:'+eq)
    train_dataset, test_dataset, min_data, scale_ = make_dataset(f, x_train, x_test)

    trained_model = train(model, train_dataset,
                          batch_size=batch_size,
                          epochs=1000,
                          index=args.index)

    y_train_pred = trained_model.predict(train_dataset[0])
    y_test_pred = trained_model.predict(test_dataset[0])
    # plt.figure()
    # plt.plot(train_dataset[0], train_dataset[1], '.', label='$(x_{train}, y_{train})$ no noise', color='C2')
    # plt.plot(x_train, y_train_pred, '.', label='$(x_{train}, NN(x_{train}))$', ms=3, color='C0')
    # plt.plot(test_dataset[0], test_dataset[1], '.', label='$(x_{test}, NN(x_{test}))$', color='C3')
    # plt.plot(x_test, y_test_pred, '.', label='$(x_{test}, NN(x_{test})$', ms=3, color='C1')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.legend()
    # plt.savefig('function_approx_batchsize{}.pdf'.format(batch_size))

    print('equation', eq)

    train_rmse = np.sqrt(np.mean(np.power(y_train_pred.flatten()-train_dataset[1].flatten(), 2)))
    print('train_rmse', train_rmse)

    unscaled_y_train_pred = unnormalize(y_train_pred.flatten(), min_data, scale_)
    unscaled_train_dataset = unnormalize(train_dataset[1].flatten(), min_data, scale_)
    train_unscaled_rmse = np.sqrt(np.mean(np.power(unscaled_y_train_pred-unscaled_train_dataset, 2)))
    print('train_unscaled_rmse', train_unscaled_rmse)

    test_rmse = np.sqrt(np.mean(np.power(y_test_pred.flatten()-test_dataset[1].flatten(), 2)))
    print('test_rmse', test_rmse)

    test_unscaled_y_test_pred = unnormalize(y_test_pred.flatten(), min_data, scale_)
    test_unscaled_test_dataset = unnormalize(test_dataset[1].flatten(), min_data, scale_)
    test_unscaled_rmse = np.sqrt(np.mean(np.power(test_unscaled_y_test_pred-test_unscaled_test_dataset, 2)))
    print('test_unscaled_rmse', test_unscaled_rmse)

    os.makedirs('rmse', exist_ok=True)

    with open('rmse/{}.txt'.format(args.index), 'w') as f:
        f.write('train_rmse,train_unscaled_rmse,test_rmse,test_unscaled_rmse\n')
        f.write('{},{},{},{}'.format(train_rmse, train_unscaled_rmse, test_rmse, test_unscaled_rmse))
