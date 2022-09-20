from script.model import Model, DataSet
from script.printer import Printer

import json
import matplotlib.pyplot as plt
import math
import numpy as np
import os

from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def compare(models, data_set, run_params):

    # Prepare out folder

    graph_folder = os.path.join(data_set.path_prefix(), 'combined_graphs_') + str(np.random.random())[2:]
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)
    for model in models:
        with open(os.path.join(graph_folder, '{}_params.json'.format(model['name'])), 'w') as out_file:
            json.dump(model['params'], out_file)
    with open(os.path.join(graph_folder, 'default_run.json'), 'w') as out_file:
        json.dump(run_params, out_file)

    # Load existing prediction or load or train the model

    try:
        for model in models:
            model['prefix'] = os.path.join(data_set.path_prefix(),
                                           Model.model_run_params_prefix(model['params'], run_params))
            model['pred'] = np.load(model['prefix'] + 'pred.npy')
            model['x_true'] = np.load(model['prefix'] + 'x_true.npy')
    except FileNotFoundError as e:
        Printer.print_log(True, 'models not found, {}'.format(e))
        exit(1)
    params = models[0]['params']

    # Calculate errors, horizons and variances

    data = data_set.data
    ics = np.arange(run_params['start_i_c'],
                    run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'],
                    run_params['i_c_distance'])
    name_map = ['hard', 'medium', 'easy']
    rnd_ics = np.random.choice(run_params['num_i_c'], 100, replace=False)
    threshold = models[0]['params']['n_inp'] * 0.0375

    Printer.add(True, 'Calculating horizons')

    for model in models:

        Printer.print_log(True, model['name'])
        model['errors'] = np.zeros((run_params['num_i_c'], run_params['pred_len']))
        model['horizon'] = np.zeros((run_params['num_i_c'],))

        for count, init_cond in enumerate(ics):
            comp_data = data[:, init_cond:init_cond + run_params['pred_len']]
            e = np.linalg.norm(model['pred'][count] - comp_data, axis=0) / model['x_true']
            model['errors'][count, :] = e
            model['horizon'][count] = np.argmax(e > threshold) if (e > threshold).any() else len(e) - 1
        model['error'] = np.average(model['errors'], axis=0)
        model['gmm'] = GaussianMixture(n_components=3).fit(model['horizon'].reshape(-1, 1))
        model['hor_class'] = model['gmm'].predict(model['horizon'].reshape(-1, 1))
        means = [mean for mean in model['gmm'].means_[:, 0]]
        class_map = np.argsort(means)
        model['hor_class'] = [name_map[np.argmax(class_map == bin_class)] for bin_class in model['hor_class']]
        model['hor_per_diff'] = [
            np.array([x for i, x in enumerate(model['horizon']) if model['hor_class'][i] == 'hard']),
            np.array([x for i, x in enumerate(model['horizon']) if model['hor_class'][i] == 'medium']),
            np.array([x for i, x in enumerate(model['horizon']) if model['hor_class'][i] == 'easy'])
        ]
        model['gmm_middles'] = \
            [(max(hor_class) + min(hor_class)) / 2 / data_set.dt_per_mtu for hor_class in model['hor_per_diff']]
        model['bin_width'] = \
            [(max(hor_class) - min(hor_class)) / data_set.dt_per_mtu for hor_class in model['hor_per_diff']]
        adjusted_ics = [
            np.array([int(ic - len([1 for i, x in enumerate(model['hor_class']) if x != 'hard' and i < ic]))
                      for ic in rnd_ics if model['hor_class'][ic] == 'hard']),
            np.array([int(ic - len([1 for i, x in enumerate(model['hor_class']) if x != 'medium' and i < ic]))
                      for ic in rnd_ics if model['hor_class'][ic] == 'medium']),
            np.array([int(ic - len([1 for i, x in enumerate(model['hor_class']) if x != 'easy' and i < ic]))
                      for ic in rnd_ics if model['hor_class'][ic] == 'easy'])
        ]
        model['errors_per_diff'] = [
            np.array([x for i, x in enumerate(model['errors']) if model['hor_class'][i] == 'hard']),
            np.array([x for i, x in enumerate(model['errors']) if model['hor_class'][i] == 'medium']),
            np.array([x for i, x in enumerate(model['errors']) if model['hor_class'][i] == 'easy'])
        ]
        model['variances'] = [
            [np.std(errors, axis=0) for errors in model['errors_per_diff']],
            [np.std(errors[:10000], axis=0) for errors in model['errors_per_diff']],
            [np.std(errors[adjusted_ics[int(i)]], axis=0) for i, errors in enumerate(model['errors_per_diff'])]
        ]
        model['errors_per_diff'] = [np.average(errors, axis=0) for errors in model['errors_per_diff']]
    print()

    # Plot average horizon lines

    fig, ax = plt.subplots()
    plt.suptitle('Mean error per model')
    x_values = np.arange(run_params['pred_len']) / data_set.dt_per_mtu

    for model in models:
        variance = np.std(model['errors'], axis=0)
        ax.plot(x_values, model['error'], label=model['name'], color=model['color'])
        ax.plot(x_values, model['error'] - variance, label='{} - std'.format(model['name']),
                color=model['color'], alpha=0.4)
        ax.plot(x_values, model['error'] + variance, label='{} - std'.format(model['name']),
                color=model['color'], alpha=0.4)

    ax.plot(x_values, [threshold for _ in range(run_params['pred_len'])], 'r--', color='black')
    ax.legend()
    ax.set_ylabel('L2')
    ax.set_xlabel('MTU')

    plt.savefig(os.path.join(graph_folder, 'error.png'), dpi=300)

    # Plotting prediction horizons

    width = max(run_params['num_i_c'] / 300, 10)
    fig, ax = plt.subplots(params['n_inp'] + 1, figsize=(width, 1.4 * (params['n_inp'] + 1)), sharex=True)
    plt.suptitle('Prediction horizon')
    x_values_var = np.arange(run_params['num_i_c'] + run_params['pred_len']) / data_set.dt_per_mtu
    x_values = np.arange(run_params['num_i_c']) / data_set.dt_per_mtu

    for var in range(params['n_inp']):
        data_from = run_params['start_i_c']
        data_to = run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'] + run_params['pred_len']
        data_skip = run_params['i_c_distance']
        ax[var].plot(
            x_values_var,
            data[var, data_from:data_to:data_skip],
            'black', label='Variable: {}/{}'.format(var + 1, params['n_inp']))
        ax[var].set_xlim([0, x_values_var[-1]])
        ax[var].legend(loc=1)

    hor_id = params['n_inp']
    for model in models:
        ax[hor_id].plot(x_values, model['horizon'] / data_set.dt_per_mtu, label=model['name'], color=model['color'])
    ax[hor_id].set_xlabel('initial condition (shift from start in MTU)')
    ax[hor_id].set_ylabel('MTU')
    ax[hor_id].set_xlim([0, x_values_var[-1]])
    ax[hor_id].legend(loc=1)

    plt.savefig(os.path.join(graph_folder, 'pred_horizons.png'), dpi=300)

    # Plot sorted prediction horizons

    width = max(run_params['num_i_c'] / 300, 10)
    fig, ax = plt.subplots(len(models), sharex=True, figsize=(width, 1.4 * len(models)))
    plt.suptitle('Sorted prediction horizon')
    x_values_var = np.arange(run_params['num_i_c']) / data_set.dt_per_mtu
    x_values = np.arange(run_params['num_i_c']) / data_set.dt_per_mtu
    sorted_ind = np.argsort(models[0]['horizon'])[::-1]

    for i, model in enumerate(models):
        ax[i].plot(x_values, model['horizon'][sorted_ind] / data_set.dt_per_mtu,
                   label=model['name'], color=model['color'])
        ax[i].legend(loc=1)
        ax[i].set_ylim([0, max(model['horizon']) / data_set.dt_per_mtu])
    ax[-1].set_ylabel('MTU')
    ax[-1].set_xlim([0, x_values_var[-1]])

    plt.xticks(np.arange(0, run_params['num_i_c'] / data_set.dt_per_mtu + 0.1, 1.0))

    plt.savefig(os.path.join(graph_folder, 'sorted_pred_horizons.png'), dpi=300)

    # Plot histograms

    plt.figure()
    fig, ax = plt.subplots(len(models) * 2, sharex=True, figsize=(8, 8))
    plt.suptitle('Prediction horizon histogram')
    curve_points = 1000

    for i, model in enumerate(models):

        groups = [
            np.array([(x, i) for i, x in enumerate(model['horizon']) if models[0]['hor_class'][i] == 'hard']),
            np.array([(x, i) for i, x in enumerate(model['horizon']) if models[0]['hor_class'][i] == 'medium']),
            np.array([(x, i) for i, x in enumerate(model['horizon']) if models[0]['hor_class'][i] == 'easy'])
        ]

        groups = np.array([
            np.array([
                len([x for x, i in group if model['hor_class'][int(i)] == 'hard']),
                len([x for x, i in group if model['hor_class'][int(i)] == 'medium']),
                len([x for x, i in group if model['hor_class'][int(i)] == 'easy'])
            ]) for group in groups
        ])

        # Non-scaled graph

        ax[2 * i].bar(model['gmm_middles'], groups[0], color='red', width=model['bin_width'])
        ax[2 * i].bar(model['gmm_middles'], groups[1], bottom=groups[0], color='yellow', width=model['bin_width'])
        ax[2 * i].bar(model['gmm_middles'], groups[2], bottom=[i + j for i, j in zip(groups[0], groups[1])],
                      color='green', width=model['bin_width'])
        ax[2 * i].set_title(model['name'])

        hist_area = np.sum(model['bin_width'] * np.sum(groups, axis=1))
        for j in range(3):
            mu = model['gmm'].means_[j]
            sigma = math.sqrt(model['gmm'].covariances_[j])
            x = np.linspace(min(model['horizon']), max(model['horizon']), curve_points)
            curve = norm.pdf(x, mu, sigma)
            curve = curve * hist_area / sum(curve) / sum(model['bin_width']) * curve_points * model['gmm'].weights_[j]
            ax[2 * i].plot(x / data_set.dt_per_mtu, curve, color='orange')

        total_curve = np.exp(model['gmm'].score_samples(np.linspace(min(model['horizon']), max(model['horizon']),
                                                                    curve_points).reshape(-1, 1)))
        total_curve /= sum(total_curve)
        total_curve *= hist_area / sum(model['bin_width']) * curve_points
        ax[2 * i].plot(np.linspace(min(model['horizon']), max(model['horizon']), curve_points) / data_set.dt_per_mtu,
                       total_curve, color='blue')

        # Scaled graph

        totals = [i + j + k for i, j, k in zip(groups[0], groups[1], groups[2])]
        groups = np.array([
            np.array([i / j * 100 if j != 0 else 0 for i, j in zip(group, totals)]) for group in groups
        ])

        ax[2 * i + 1].bar(model['gmm_middles'], groups[0], color='red', width=model['bin_width'])
        ax[2 * i + 1].bar(model['gmm_middles'], groups[1], bottom=groups[0], color='yellow', width=model['bin_width'])
        ax[2 * i + 1].bar(model['gmm_middles'], groups[2], bottom=[i + j for i, j in zip(groups[0], groups[1])],
                          color='green', width=model['bin_width'])
        ax[2 * i + 1].set_title('{} - proportional'.format(model['name']))

    plt.xlim(xmin=0.0)
    plt.xlabel('MTU')

    plt.savefig(os.path.join(graph_folder, 'pred_horizons_histogram.png'), dpi=300)

    # Plot average horizon lines with variances

    x_values = np.arange(run_params['pred_len']) / data_set.dt_per_mtu

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle('Mean error with variances per model')

    def plot_variances(i, title):

        for j in range(3):
            max_horizon = 0
            for model in models:
                axs[i, j].plot(x_values, model['errors_per_diff'][j], label=model['name'], color=model['color'])
                axs[i, j].plot(x_values, model['errors_per_diff'][j] - model['variances'][i][j],
                               label='{} - std'.format(model['name']), color=model['color'], alpha=0.4)
                axs[i, j].plot(x_values, model['errors_per_diff'][j] + model['variances'][i][j],
                               label='{} + std'.format(model['name']), color=model['color'], alpha=0.4)
                max_horizon = max(max_horizon,
                                  np.argmax(model['errors_per_diff'][j] - model['variances'][i][j] > threshold))

            axs[i, j].plot(x_values, [threshold for _ in range(run_params['pred_len'])], 'r--', color='black')
            axs[i, j].legend(loc=1)
            axs[i, j].set_title('{} - {}'.format(title, name_map[j]))
            axs[i, j].set_xlabel('L2')
            axs[i, j].set_xlabel('MTU')
            axs[i, j].set_xlim([0, max_horizon / data_set.dt_per_mtu])
            axs[i, j].set_ylim([0, 2 * threshold])

    plot_variances(0, 'all')
    plot_variances(1, '10000')
    plot_variances(2, '100')

    plt.savefig(os.path.join(graph_folder, 'error_variances.png'), dpi=300)


def main():

    data_set = DataSet('lorenz_96', 200, 1)

    # Load run parameters

    run_params = Model.default_run_params()

    # Define runs and load params

    models_set = set()

    models1 = [
        {
            'name': 'esn',
            'params': Model.default_model_params('esn'),
            'color': 'orange'
        },
        {
            'name': 'esn_tr_skip',
            'params': Model.default_model_params('esn'),
            'color': 'blue'
        }
    ]
    models1[1]['params']['tr_skip'] = 300000

    models2 = [
        {
            'name': 'esn',
            'params': Model.default_model_params('esn'),
            'color': 'orange'
        },
        {
            'name': 'esn_seeds',
            'params': Model.default_model_params('esn'),
            'color': 'blue'
        }
    ]
    models2[1]['params']['win_seed'] = 101
    models2[1]['params']['a_seed'] = 201

    models3 = [
        {
            'name': 'esn',
            'params': Model.default_model_params('esn'),
            'color': 'orange'
        },
        {
            'name': 'esn_radius',
            'params': Model.default_model_params('esn'),
            'color': 'blue'
        }
    ]
    models3[1]['params']['radius'] = 0.9

    models4 = [
        {
            'name': 'esn',
            'params': Model.default_model_params('esn'),
            'color': 'orange'
        },
        {
            'name': 'lstm',
            'params': Model.default_model_params('lstm'),
            'color': 'blue'
        }
    ]

    models5 = [
        {
            'name': 'lstm',
            'params': Model.default_model_params('lstm'),
            'color': 'orange'
        },
        {
            'name': 'lstm_tr_skip',
            'params': Model.default_model_params('lstm'),
            'color': 'blue'
        }
    ]
    models5[1]['params']['tr_skip'] = 300000

    models6 = [
        {
            'name': 'esn',
            'params': Model.default_model_params('esn'),
            'color': 'orange'
        },
        {
            'name': 'ensemble',
            'params': Model.default_model_params('ensemble_1'),
            'color': 'blue'
        }
    ]

    chosen_models = models3

    print(chosen_models)

    compare(chosen_models, data_set, run_params)


if __name__ == '__main__':
    main()
