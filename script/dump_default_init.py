# Define and dump default init data

import json
import numpy as np

from script.model import DataSet


res_params = {
    'n_inp': 8,
    "tr_skip": 0,
    'tr_len': 200000,
    'radius': 0.1,
    'degree': 3,
    'sigma': 0.5,
    'beta': 0.0001,
    'win_seed': 100,
    'a_seed': 200,
    'fun': 'tanh'
}

res_params['size'] = int(np.floor(5000 / res_params['n_inp']) * res_params['n_inp'])


run_params = {
    'warm_up_size': 4,
    'num_i_c': 1000,
    'pred_len': 1000,
    'start_i_c': res_params['tr_len'] + 100000
}

spread_ics_over_data_set = False

if spread_ics_over_data_set:
    data_set = DataSet('lorenz_96', 200, 1)
    data = data_set.data
    run_params['i_c_distance'] = \
        int(
            (data.shape[1] - run_params['pred_len'] - run_params['start_i_c']) /
            run_params['num_i_c']
        )
else:
    run_params['i_c_distance'] = 1


dump_obj = {
    'res_params': res_params,
    'run_params': run_params
}

with open('default_esn.json', 'w') as out_file:
    json.dump(dump_obj, out_file)
