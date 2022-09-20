import numpy as np
import os
from script.model import Model, DataSet

print('Dumping list of prediction horizons to file')

data_set = DataSet('lorenz_96', 200, 1)
data = data_set.data

model_params = Model.default_model_params('esn')
run_params = Model.default_run_params()
print(model_params)
print(run_params)

path_prefix = os.path.join(data_set.path_prefix() + Model.model_run_params_prefix(model_params, run_params))

predictions_path = path_prefix + 'pred.npy'
predictions = np.load(predictions_path)

x_true_path = path_prefix + 'x_true.npy'
x_true = np.load(x_true_path)

threshold = model_params['n_inp'] * 0.0375
horizons = np.zeros((run_params['num_i_c'], ))
for i, prediction in enumerate(predictions):
    comp_data = data[:, run_params['start_i_c'] + i:run_params['start_i_c'] + i + run_params['pred_len']]
    e = np.linalg.norm(prediction - comp_data, axis=0) / x_true
    horizons[i] = np.argmax(e > threshold) if (e > threshold).any() else len(e) - 1

with open(path_prefix + 'horizons.csv', 'w') as csv_file:
    for i in range(run_params['num_i_c']):
        csv_file.write('{}, {}\n'.format(run_params['start_i_c'] + i, horizons[i]))
