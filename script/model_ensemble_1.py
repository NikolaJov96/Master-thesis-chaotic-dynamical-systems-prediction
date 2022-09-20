from script.model import Model
from script.model_esn import ModelEsn
from script.model_lstm import ModelLstm

from script.printer import Printer

import json
import numpy as np
import os


# Model class representing ensemble of ESN-s

class ModelEnsemble1(Model):

    # Initialize model parameters and variables

    def __init__(self, models_params, data_set_path):
        super().__init__(data_set_path)

        self.n_inp = models_params["n_inp"]

        self.models = []
        self.model_types = []

        for model_params in models_params['models']:
            if model_params['type'] == 'esn':
                self.models.append(ModelEsn(model_params['params'], data_set_path))
            elif model_params['type'] == 'lstm':
                self.models.append(ModelLstm(model_params['params'], data_set_path))
            self.model_types.append(model_params['type'])

    def params_to_dict(self):

        model_dict = {"n_inp": self.n_inp, "models": []}
        for model_type, model in zip(self.model_types, self.models):
            model_dict['models'].append({
                'type': model_type,
                'params': model.params_to_dict()
            })
        return model_dict

    def load_or_train(self, data, verbose):

        for i, model in enumerate(self.models):
            Printer.add(verbose, 'Ensemble {}/{}'.format(i + 1, len(self.models)))
            model.load_or_train(data, verbose)
            Printer.rem(verbose)
        folder_name = self.dict_to_os_path_with_data(self.params_to_dict())
        print('qwer', folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        with open(os.path.join(folder_name, 'params.json'), 'w') as out_file:
            json.dump(self.params_to_dict(), out_file)

    def predict(self, warm_up_input, run_params, pos_inside_run, verbose):

        predictions = []
        for i, model in enumerate(self.models):
            Printer.add(verbose, 'Ensemble {}/{}'.format(i + 1, len(self.models)))
            predictions.append(model.predict(warm_up_input, run_params, pos_inside_run, verbose))
            Printer.rem(verbose)
        predictions = np.array(predictions)
        output = np.average(predictions, axis=0)

        return output
