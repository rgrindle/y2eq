"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 15, 2020

PURPOSE: Train a NN to do function approximation
         for a single function.

NOTES:

TODO:
"""
from architecture_1d import model

from tensorflow.keras.callbacks import ModelCheckpoint

import os


def train(model, dataset, batch_size, epochs):
    model_name = 'function_approximation'
    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('models', 'weights_'+model_name),
                                 monitor='val_loss')

    model.compile(optimizer='adam', loss='mse')
    # NOTE: y gets padded inside model as input.
    history = model.fit(dataset[0], dataset[1],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_cb, weights_cb])

    # plt.figure()
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='val')
    # plt.legend()
    # plt.show()

    return model


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    def normalize(data, min_data=None, scale_=None):
        if min_data is None or scale_ is None:
            min_data = np.min(data)
            max_data = np.max(data)
            scale_ = 1./(max_data-min_data)
        return (data-min_data)*scale_, min_data, scale_

    batch_size = 1

    # make a dataset
    np.random.seed(0)
    x = np.random.uniform(-1, 1, 300)
    f = lambda x: 3*x*np.sin(5*x)+7
    y_train, min_data, scale_ = normalize(f(x))
    train_dataset = [x, y_train]
    x_test = np.linspace(-1, 1, 30)
    y_test = normalize(f(x_test), min_data, scale_)[0]
    test_dataset = [x_test, y_test]
    print(test_dataset[1])

    trained_model = train(model, train_dataset,
                          batch_size=batch_size,
                          epochs=300)

    # result = trained_model.evaluate(*test_dataset)
    # print(result)
    y_train_pred = trained_model.predict(x)
    y_test_pred = trained_model.predict(x_test)
    plt.figure()
    plt.plot(x, y_train_pred, '.', label='train')
    plt.plot(x_test, y_test_pred, '.', label='test')
    plt.plot(train_dataset[0], train_dataset[1], '-', label='actual')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    plt.savefig('function_approx_batchsize{}.pdf'.format(batch_size))
