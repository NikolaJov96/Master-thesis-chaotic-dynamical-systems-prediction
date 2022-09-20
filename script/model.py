from abc import abstractmethod

import hashlib
import json
import numpy as np
import os
import pandas as pd


# Class representing data set

class DataSet:

    # Load data set

    def __init__(self, file_name, dt_per_mtu, skip):

        self.file_name = file_name
        self.dt_per_mtu = dt_per_mtu
        self.skip = skip
        try:
            print('Loading {}... '.format(file_name), end='')
            self.data = np.transpose(np.array(pd.read_csv(
                os.path.join('data_sets', '{}.csv'.format(file_name)), header=None
            )))[:, ::skip]
            print('Done')
        except FileNotFoundError:
            print('Error, file not found')
            exit(1)

    # Create dat set folder and return it's name

    def path_prefix(self):

        name = '{}, {}, {}'.format(self.file_name, self.dt_per_mtu, self.skip)
        if not os.path.exists(name):
            os.mkdir(name)
        return name


# Class representing interface for any used prediction model

class Model:

    def __init__(self, data_set_path):

        self.data_set_path = data_set_path

    # Generate folder or file name from dict data, with data set folder

    def dict_to_os_path_with_data(self, model_params):

        return os.path.join(self.data_set_path, Model.dict_to_os_path(model_params))

    # Generate specific file prefix using model_params and run_params, with data set folder

    def model_run_param_prefix_with_data(self, model_params, run_params):

        return os.path.join(self.data_set_path, Model.model_run_params_prefix(model_params, run_params))

    # Generate dictionary with model parameters

    @abstractmethod
    def params_to_dict(self):
        pass

    # Try to load trained model, if it does not exist train

    @abstractmethod
    def load_or_train(self, data, verbose):
        pass

    # Execute free prediction sing warm-up

    @abstractmethod
    def predict(self, warm_up_input, run_params, pos_inside_run, verbose):
        pass

    # Load default params of requested model from json file

    @staticmethod
    def default_model_params(model_name):
        with open('default_{}.json'.format(model_name), 'r') as model_params_file:
            return json.load(model_params_file)

    # Load default simulation parameters form json file

    @staticmethod
    def default_run_params():
        with open('default_run.json', 'r') as run_params_file:
            return json.load(run_params_file)

    # Generate folder or file name from dict data

    @staticmethod
    def dict_to_os_path(res_params):

        path = str(list(res_params.values()))[1:-1].replace("'", "").replace(".", ",").\
            replace("{", "").replace("}", "").replace(":", "")
        if len(path) > 70:
            path = 'ensemble_{}'.format(hashlib.md5(path.encode('utf-8')).hexdigest())
        return path

    # Generate specific file prefix using res_params and run_params

    @staticmethod
    def model_run_params_prefix(model_params, run_params):

        return os.path.join(Model.dict_to_os_path(model_params), Model.dict_to_os_path(run_params) + ', ')
