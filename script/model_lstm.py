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


# Model class representing LSTM with variable look-back

class ModelLstm(Model):

    # Initialize model parameters and variables

    def __init__(self, model_params, data_set_path):
        super().__init__(data_set_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)

        # Parameters
        self.n_inp = model_params['n_inp']
        self.tr_skip = model_params["tr_skip"]
        self.look_back = model_params['look_back']
        self.tr_len = model_params['tr_len']
        self.n_hidden = model_params['n_hidden']
        self.do_basis_exp = model_params['do_basis_exp']
        self.epochs = model_params['epochs']
        self.batch_size = model_params['batch_size']
        self.shuffle = model_params['shuffle']

        # State
        self.model = None
        self.predictions = {}

    def params_to_dict(self):
        return {
            'n_inp': self.n_inp,
            'tr_skip': self.tr_skip,
            'look_back': self.look_back,
            'tr_len': self.tr_len,
            'n_hidden': self.n_hidden,
            'do_basis_exp': self.do_basis_exp,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }

    # Convert simple to stacked data for given LSTM look back

    def stack_data(self, data):
        whole_data_len = data.shape[1]
        data_points = whole_data_len - self.look_back + 1

        x_temp = np.zeros((self.look_back, self.n_inp, data_points))
        for i in range(self.look_back):
            x_temp[i, :] = data[:, i:data_points + i]

        x = x_temp[0]
        for i in range(self.look_back - 1):
            x = np.vstack([x, x_temp[i + 1]])

        x = np.transpose(x)
        return x.reshape((data_points, self.look_back, self.n_inp))

    # Generate model and train Wout

    def train(self, data, verbose):

        Printer.add(verbose, 'training')

        self.model = Sequential()
        # self.model.add(LSTM(self.n_hidden, input_shape=(self.look_back, self.n_inp)))
        self.model.add(CuDNNLSTM(self.n_hidden, input_shape=(self.look_back, self.n_inp)))

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
        # The last data point is needed as a label for the last stacked training set entry?
        x = self.stack_data(data[:, :self.tr_len - 1])
        y = np.transpose(data[:, self.look_back:self.tr_len])

        # Fit network

        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=self.shuffle)
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

        stacked_input = self.stack_data(warm_up_input)

        for i in range(run_params['pred_len']):
            pred = self.model.predict(stacked_input)
            output[i, :] = pred
            stacked_input[0, :-1, :] = stacked_input[0, 1:, :]
            stacked_input[0, -1, :] = pred

        Printer.rem(verbose)
        return output.T
