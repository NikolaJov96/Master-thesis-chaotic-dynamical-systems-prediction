from script.model import Model

from script.printer import Printer

import json
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf


# Model class representing LSTM with no look-back

class ModelLstm2(Model):

    # Initialize model parameters and variables

    def __init__(self, model_params, data_set_path):
        super().__init__(data_set_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)

        # Parameters
        self.n_inp = model_params['n_inp']
        self.tr_skip = model_params["tr_skip"]
        self.warm_up_size = model_params['warm_up_size']
        self.tr_len = model_params['tr_len']
        self.n_hidden = model_params['n_hidden']
        self.do_basis_exp = model_params['do_basis_exp']
        self.epochs = model_params['epochs']
        self.batch_size = model_params['batch_size']
        self.batches_per_epoch = model_params['batches_per_epoch']
        self.shuffle = model_params['shuffle']

        # State
        self.model = None
        self.predictions = {}

    def params_to_dict(self):
        return {
            'n_inp': self.n_inp,
            'tr_skip': self.tr_skip,
            'warm_up_size': self.warm_up_size,
            'tr_len': self.tr_len,
            'n_hidden': self.n_hidden,
            'do_basis_exp': self.do_basis_exp,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'batches_per_epoch': self.batches_per_epoch,
            'shuffle': self.shuffle,
        }

    # Generator function for creating random batches of training-data.

    def batch_generator(self, data):

        # Infinite loop.
        while True:
            # Allocate a new array for the batch of input-signals.
            x_shape = (self.batch_size, self.warm_up_size, self.n_inp)
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            # Allocate a new array for the batch of output-signals.
            y_shape = (self.batch_size, self.warm_up_size, self.n_inp)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            # Fill the batch with random sequences of data.
            for i in range(self.batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                idx = np.random.randint(self.tr_len - 1 - self.warm_up_size)

                # Copy the sequences of data starting at this index.
                x_batch[i] = data[idx:idx + self.warm_up_size]
                y_batch[i] = data[idx + 1:idx + 1 + self.warm_up_size]

            yield (x_batch, y_batch)

    # Calculate the Mean Squared Error between y_true and y_pred, ignoring "warm-up" part of the sequences.
    @staticmethod
    def loss_mse_warm_up(y_true, y_pred, run_params):

        # The shape of both input tensors are:
        # [batch_size, sequence_length, num_y_signals].

        # Ignore the "warm-up" parts of the sequences
        # by taking slices of the tensors.
        y_true_slice = y_true[:, run_params['warm_up_size']:, :]
        y_pred_slice = y_pred[:, run_params['warm_up_size']:, :]

        # These sliced tensors both have this shape:
        # [batch_size, sequence_length - warm-up_steps, num_y_signals]

        # Calculate the MSE loss for each value in these tensors.
        # This outputs a 3-rank tensor of the same shape.
        loss = tf.losses.mean_squared_error(labels=y_true_slice, predictions=y_pred_slice)

        # Keras may reduce this across the first axis (the batch)
        # but the semantics are unclear, so to be sure we use
        # the loss across the entire tensor, we reduce it to a
        # single scalar with the mean function.
        loss_mean = tf.reduce_mean(loss)

        return loss_mean

    # Generate model and train Wout

    def train(self, data, verbose):

        Printer.add(verbose, 'training')

        self.model = Sequential()
        # self.model.add(LSTM(self.n_hidden, return_sequences=True, input_shape=(None, self.n_inp, )))
        self.model.add(CuDNNLSTM(self.n_hidden, return_sequences=True, input_shape=(None, self.n_inp, )))

        if self.do_basis_exp:
            delinearize_layer = Dense(self.n_hidden)
            built_layer = delinearize_layer(Input((self.n_hidden,)))
            built_layer.trainable = False
            weights = delinearize_layer.get_weights()
            weights[0] = np.zeros((self.n_hidden, self.n_hidden))
            for i in range(0, self.n_hidden, 2):
                weights[0][i, i] = 1
            for i in range(1, self.n_hidden, 2):
                weights[0][i, i] = 1
                weights[0][i, i - 1] = 1
            delinearize_layer.set_weights(weights)
            self.model.add(delinearize_layer)

        self.model.add(Dense(self.n_inp))
        self.model.compile(loss='mse', optimizer='adam')

        # Prepare data

        # Call stack data and pass the whole training set except for the last data point
        # The last data point is needed as a label for the last stacked training set entry
        # x = data[:, :self.tr_len - 1].T
        # y = np.transpose(data[:, 1:self.tr_len])

        # Fit network

        # self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=self.shuffle)
        self.model.fit_generator(
            generator=self.batch_generator(data.T),
            epochs=self.epochs,
            steps_per_epoch=self.batches_per_epoch,
            callbacks=[],
            verbose=1
        )
        self.model._make_predict_function()

        Printer.rem(verbose)

    # Load reservoir if it exists, else train it

    def load_or_train(self, data, verbose):

        folder_name = self.dict_to_os_path_with_data(self.params_to_dict())
        try:

            Printer.print_log(verbose, 'Loading LSTM model... ')
            self.model = load_model(os.path.join(folder_name, 'lstm_model.h5'))
            self.model._make_predict_function()
            Printer.print_log(verbose, 'Loading LSTM model... Done')

        except OSError:

            self.train(data[:, self.tr_skip:self.tr_skip + self.tr_len], verbose)

            Printer.print_log(verbose, 'Saving LSTM model... ')
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            with open(os.path.join(folder_name, 'params.json'), 'w') as out_file:
                json.dump(self.params_to_dict(), out_file)
            self.model.save(os.path.join(folder_name, 'lstm_model.h5'))
            Printer.print_log(verbose, 'Saving LSTM model... Done')

    # Predict function

    def predict(self, warm_up_input, run_params, pos_inside_run, verbose):

        if str(run_params) in self.predictions:
            return self.predictions[str(run_params)][pos_inside_run]
        else:
            try:
                file_prefix = os.path.join(self.data_set_path, Model.model_run_params_prefix(self.params_to_dict(), run_params))
                self.predictions = np.load(file_prefix + 'pred.npy')
                return self.predictions[pos_inside_run]
            except OSError:
                pass

        Printer.add(verbose, 'predicting')

        output = np.zeros((run_params['pred_len'], self.n_inp))

        output[0, :] = self.model.predict(warm_up_input.T.reshape(1, run_params['warm_up_size'], self.n_inp))[0, -1, :]

        for i in range(1, run_params['pred_len']):
            output[i, :] = self.model.predict(output[i - 1].reshape(1, 1, self.n_inp))

        Printer.rem(verbose)
        return output.T
