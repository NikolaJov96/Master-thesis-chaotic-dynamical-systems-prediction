from script.model import Model, DataSet
from script.model_esn import ModelEsn

import math
import numpy as np
import os


def execute_training(params, data_set):

    model = ModelEsn(params, data_set.path_prefix())
    model.load_or_train(data_set.data, True)
    return model


def execute_prediction(data_set, model_params, run_params, model):

    file_prefix = os.path.join(data_set.path_prefix(), Model.model_run_params_prefix(model_params, run_params))

    if not os.path.exists(file_prefix + 'pred.npy'):

        ics = range(run_params['start_i_c'],
                    run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'],
                    run_params['i_c_distance'])

        predictions = np.zeros((run_params['num_i_c'], model_params['n_inp'], run_params['pred_len']))

        for count, init_cond in enumerate(ics):

            warm_up_input = data_set.data[:, init_cond - run_params['warm_up_size']:init_cond]
            prediction = model.predict(warm_up_input, run_params, count, True)
            predictions[count, :] = prediction

        np.save(file_prefix + 'pred.npy', np.array(predictions))


def gen_params(default_params, sweep_params, ind):

    act_params = default_params.copy()
    div_step = 1
    for entry in sweep_params:
        range_size = len(entry['ranges'])
        for param_ind in range(len(entry['params'])):
            act_params[entry['params'][int(param_ind)]] = entry['ranges'][int(int(ind / div_step) % range_size)][
                param_ind]
        div_step *= range_size
    return act_params


def execute_bach(i, per_bach, total_runs, sweep_params, data_set, default_params, run_params, do_predict):

    start_ind = i * per_bach
    end_ind = min((i + 1) * per_bach, total_runs) - 1

    act_params_set = []
    models = []

    for ind in range(start_ind, end_ind + 1):

        print('{}/{}'.format(ind + 1, total_runs))
        act_params = gen_params(default_params, sweep_params, ind)
        act_params_set.append(act_params)
        model = execute_training(act_params, data_set)
        models.append(model)

    if do_predict:
        for model, act_params in zip(models, act_params_set):
            execute_prediction(data_set, act_params, run_params, model)


def main():

    data_set = DataSet('lorenz_96', 200, 1)

    # Load run parameters

    run_params = Model.default_run_params()

    default_params = Model.default_model_params('esn')
    print("Default params:")
    print(default_params)

    sweep_params = [
        {
            'params': ['size'],
            'ranges': [[1000], [2250], [3500], [4750], [6000]]
        },
        {
            'params': ['tr_len'],
            'ranges': [[25000], [50000], [100000], [200000], [400000]]
        },
        {
            'params': ['win_seed', 'a_seed'],
            'ranges': [[100, 200], [101, 201]]
        },
        {
            'params': ['radius'],
            'ranges': [[0.1], [0.5], [0.9]]
        },
        {
            'params': ['degree'],
            'ranges': [[2], [3], [5]]
        },
        {
            'params': ['sigma'],
            'ranges': [[0.3], [0.5], [0.7]]
        },
        {
            'params': ['beta'],
            'ranges': [[0.00005], [0.0001], [0.0002]]
        },
    ]

    sweep_params = [
        {
            'params': ['size'],
            'ranges': [[500], [1000], [150]]
        },
        {
            'params': ['tr_len'],
            'ranges': [[20000], [40000], [60000]]
        },
    ]

    total_runs = np.prod([len(entry['ranges']) for entry in sweep_params])
    print('Total runs: {}'.format(total_runs))

    batches = 10
    per_bach = math.ceil(total_runs / batches)
    do_predict = True

    for i in range(batches):
        execute_bach(i, per_bach, total_runs, sweep_params, data_set, default_params, run_params, do_predict)


if __name__ == '__main__':
    main()
