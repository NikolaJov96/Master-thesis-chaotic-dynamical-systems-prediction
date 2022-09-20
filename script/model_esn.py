from script.model import Model

from script.printer import Printer

import json
import numpy as np
import os
import scipy.sparse as sparse


# Model class representing ESN

class ModelEsn(Model):

    # Initialize model parameters and variables

    def __init__(self, model_params, data_set_path):
        super().__init__(data_set_path)

        # Parameters
        self.n_inp = model_params["n_inp"]
        self.tr_skip = model_params["tr_skip"]
        self.tr_len = model_params["tr_len"]
        self.radius = model_params["radius"]
        self.degree = model_params["degree"]
        self.sigma = model_params["sigma"]
        self.beta = model_params["beta"]
        self.win_seed = model_params["win_seed"]
        self.a_seed = model_params["a_seed"]
        self.fun = model_params["fun"]
        self.size = model_params["size"]

        # State
        self.w_in = None
        self.a = None
        self.a_sparse = None
        self.w_out = None
        self.predictions = {}

        # Compute reservoir state update non-linearity
        if self.fun == 'tanh':
            self.r_s_u_n_l = np.tanh
        elif self.fun == 'norm':
            self.r_s_u_n_l = lambda vector: vector / np.linalg.norm(vector)
        else:
            print()
            print('Unknown reservoir state update non-linearity function')
            exit(1)

    def params_to_dict(self):

        return {
            'n_inp': self.n_inp,
            'tr_skip': self.tr_skip,
            'tr_len': self.tr_len,
            'radius': self.radius,
            'degree': self.degree,
            'sigma': self.sigma,
            'beta': self.beta,
            'win_seed': self.win_seed,
            'a_seed': self.a_seed,
            'fun': self.fun,
            'size': self.size,
        }

    # Generate reservoir sparse matrix

    def generate_reservoir(self, verbose, a_sparse=None):

        Printer.print_log(verbose, 'generating reservoir')
        if a_sparse is None:
            sparsity = self.degree / float(self.size)
            np.random.seed(self.a_seed)
            a_sparse = sparse.rand(self.size, self.size, density=sparsity)
        a = a_sparse.todense()
        values = np.linalg.eigvals(a)
        e = np.max(np.abs(values))
        a = (a / e) * self.radius
        return a, a_sparse

    # Generate the vector of reservoir states for each training data point and return as a matrix

    def reservoir_warm_up_history(self, input_data, verbose):

        Printer.add(verbose, 'generating reservoir state history')
        Printer.print_log(verbose)
        states = np.zeros((self.size, self.tr_len))
        for i in range(self.tr_len - 1):
            if i % int((self.tr_len - 1) / 100) == 0 or i == self.tr_len - 2:
                Printer.print_log(verbose, '{}%'.format(int(i * 100 / (self.tr_len - 1))))
            states[:, i + 1] = self.r_s_u_n_l(np.dot(self.a, states[:, i]) + np.dot(self.w_in, input_data[:, i]))
        Printer.rem(verbose)
        return states

    # Delinearize matrix of inputs by multiplying every even element by its predecessor (introduce basis expansion)
    @staticmethod
    def basis_expansion_matrix(in_arr, verbose):

        Printer.print_log(verbose, 'introducing basis expansion')
        out_arr = in_arr.copy()
        for j in range(1, np.shape(in_arr)[0], 2):
            out_arr[j, :] = in_arr[j, :] * in_arr[j - 1, :]
        return out_arr

    # Generate model and train Wout

    def train(self, data, verbose):

        Printer.add(verbose, 'training')

        # Win
        Printer.print_log(verbose, 'generating Win')
        q = int(self.size / self.n_inp)
        self.w_in = np.zeros((self.size, self.n_inp))
        np.random.seed(self.win_seed)
        for i in range(self.n_inp):
            self.w_in[i * q: (i + 1) * q, i] = self.sigma * (-1 + 2 * np.random.rand(q))

        # A
        self.a, self.a_sparse = self.generate_reservoir(verbose)

        # Wout
        Printer.print_log(verbose, 'generating Wout')
        states = self.reservoir_warm_up_history(data, verbose)
        id_mat = self.beta * sparse.identity(self.size)
        expanded_states = self.basis_expansion_matrix(states, verbose)
        u = np.dot(expanded_states, expanded_states.transpose()) + id_mat
        self.w_out = np.dot(np.linalg.inv(u), np.dot(expanded_states, data.transpose())).transpose()

        Printer.rem(verbose)

    # Load reservoir if it exists, else train it

    def load_or_train(self, data, verbose):

        folder_name = self.dict_to_os_path_with_data(self.params_to_dict())
        print(folder_name)
        try:
            Printer.print_log(verbose, 'Loading reservoir... ')
            print(1)
            self.w_in = np.load(os.path.join(folder_name, 'win.npy'))
            print(2)
            self.w_out = np.load(os.path.join(folder_name, 'wout.npy'))
            print(3)
            self.a_sparse = sparse.load_npz(os.path.join(folder_name, 'a_sparse.npz'))
            print(4)
            self.a, _ = self.generate_reservoir(verbose, self.a_sparse)
            print(5)
            Printer.print_log(verbose, 'Loading reservoir... Done')

        except FileNotFoundError:

            self.train(data[:, self.tr_skip:self.tr_skip + self.tr_len], verbose)

            Printer.print_log(verbose, 'Saving reservoir... ')
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            with open(os.path.join(folder_name, 'params.json'), 'w') as out_file:
                json.dump(self.params_to_dict(), out_file)
            np.save(os.path.join(folder_name, 'win.npy'), self.w_in)
            np.save(os.path.join(folder_name, 'wout.npy'), self.w_out)
            sparse.save_npz(os.path.join(folder_name, 'a_sparse.npz'), self.a_sparse)
            Printer.print_log(verbose, 'Saving reservoir... Done')

    # Reservoir warm up function

    def reservoir_warm_up(self, warm_up_input, run_params):

        states = np.zeros((self.size, run_params['warm_up_size'] + 1))
        for i in range(run_params['warm_up_size']):
            states[:, i + 1] = self.r_s_u_n_l(np.dot(self.a, states[:, i]) + np.dot(self.w_in, warm_up_input[:, i]))
        return states[:, -1]

    # Delinearize reservoir state
    @staticmethod
    def basis_expansion_array(in_arr):

        arr_shifted = np.insert(in_arr.copy()[:-1], 0, 1)
        arr_multiply = np.multiply(in_arr, arr_shifted)
        return np.dstack((in_arr[::2], arr_multiply[1:][::2])).flatten()

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
        res_state = self.reservoir_warm_up(warm_up_input, run_params)
        output = np.zeros((self.n_inp, run_params['pred_len']))
        for i in range(run_params['pred_len']):
            if (i * 100) % run_params['pred_len'] == 0:
                Printer.print_log(verbose, '{}/{}'.format(i + 1, run_params['pred_len']))
            x_aug = ModelEsn.basis_expansion_array(res_state)
            output[:, i] = np.squeeze(np.asarray(np.dot(self.w_out, x_aug)))
            x1 = self.r_s_u_n_l(np.dot(self.a, res_state) + np.dot(self.w_in, output[:, i]))
            res_state = np.squeeze(np.asarray(x1))
        Printer.rem(verbose)
        return output
