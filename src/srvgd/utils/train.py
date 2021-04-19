"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 17, 2020

PURPOSE: Train neural networks based on achitectures
         in the architecture folder and datasets generated
         with code in the data_gathering folder.

NOTES: 12 = start, 13 = stop

TODO: make checkpoint cmd line arg, model name too
"""
from srvgd.common.cmd_line_args import get_cmd_line_args_for_datasets
from srvgd.common.save_load_dataset import load_and_format_dataset

import numpy as np  # type: ignore
import pandas as pd  # type: ignoree
from tensorflow.keras.callbacks import ModelCheckpoint

import os


def train_model(model, x, y,
                batch_size, epochs,
                model_name, checkpoint=False):
    model_cb = ModelCheckpoint(save_best_only=True,
                               filepath=os.path.join('..', '..', '..', 'models', 'model_'+model_name),
                               monitor='val_loss')

    weights_cb = ModelCheckpoint(save_best_only=True,
                                 save_weights_only=True,
                                 filepath=os.path.join('..', '..', '..', 'models', 'weights_'+model_name),
                                 monitor='val_loss')

    if not checkpoint:
        model.compile(optimizer='adam', loss='categorical_crossentropy')
    # NOTE: y gets padded inside model as input.
    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_cb, weights_cb])

    history_file = os.path.join('..', '..', '..', 'models', model_name+'_history.csv')
    if checkpoint:
        df = pd.read_csv(history_file)
        history_data = np.vstack((df.values, list(history.history.values())))
    else:
        history_data = history.history
    pd.DataFrame(history_data).to_csv(history_file, header=False, index=False)
    return model


if __name__ == '__main__':
    from tensorflow import keras

    _, dataset_name = get_cmd_line_args_for_datasets()

    checkpoint = False
    if checkpoint:
        model = keras.models.load_model(os.path.join('..', '..', '..', 'models', 'model_seq2seq_cnn_attention_model'))

    else:
        from srvgd.architecture.seq2seq_cnn_attention import get_y2eq_model
        model = get_y2eq_model()

    x, y = load_and_format_dataset(dataset_name, 'train')

    trained_model = train_model(model, x, y,
                                batch_size=128,
                                epochs=2,
                                model_name='seq2seq_cnn_attention_model_'+dataset_name,
                                checkpoint=checkpoint)
