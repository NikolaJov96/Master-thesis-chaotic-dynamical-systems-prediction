# Analise the behaviour of prediction horizon on a consecutive set of initial conditions

from script.model import Model, DataSet
from script.model_ensemble_1 import ModelEnsemble1
from script.model_esn import ModelEsn
from script.model_lstm import ModelLstm
from script.model_lstm_2 import ModelLstm2
from script.printer import Printer

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import threading
import time

from scipy.stats import norm
from sklearn.mixture import GaussianMixture


# Calculate prediction horizon

def get_horizon(e, threshold):
    return np.argmax(e > threshold) if (e > threshold).any() else len(e) - 1


# Min thread function

def thread_main(model_params, data, run_params, horizons, ind, threads, model, x_true_avg, predictions, errors):

    tasks = math.ceil(run_params['num_i_c'] / threads)
    if ind + 1 == threads:
        tasks = run_params['num_i_c'] - (threads - 1) * tasks

    ics = range(run_params['start_i_c'] + ind * tasks * run_params['i_c_distance'],
                run_params['start_i_c'] + (ind + 1) * tasks * run_params['i_c_distance'],
                run_params['i_c_distance'])

    verbose = ind == 0
    Printer.add(verbose, 'Thread: {}/{}'.format(ind + 1, threads))
    Printer.add(verbose, 'Predicting')
    Printer.print_log(verbose)

    start_time = time.time()
    threshold = model_params['n_inp'] * 0.0375
    for count, init_cond in enumerate(ics):
        Printer.add(verbose, 'IC: {}/{} - {}s'.format(count + 1, tasks, int(time.time() - start_time)))

        if model is None:
            prediction = predictions[ind * tasks + count]
        else:
            warm_up_input = data[:, init_cond - run_params['warm_up_size']:init_cond]
            prediction = model.predict(warm_up_input, run_params, ind * tasks + count, verbose)
            predictions[ind * tasks + count, :] = prediction
        Printer.print_log(verbose, 'calculating error...')
        comp_data = data[:, init_cond:init_cond + run_params['pred_len']]
        e = np.linalg.norm(prediction - comp_data, axis=0) / x_true_avg
        errors[ind * tasks + count, :] = e
        horizons[0, ind * tasks + count] = get_horizon(e, threshold)
        for inp in range(model_params['n_inp']):
            horizons[inp + 1, ind * tasks + count] = get_horizon(np.abs(prediction[inp] - comp_data[inp]), threshold)
        Printer.print_log(verbose, 'calculating error... Done')

        Printer.rem(verbose)


# Plotting prediction horizons

def plot_horizon(run_params, data, model_params, horizons, graph_folder, mtu_constant):
    Printer.print_log(True, 'Plotting prediction horizon graph... ')

    width = max(run_params['num_i_c'] / 300, 10)
    fig, ax = plt.subplots(model_params['n_inp'] + 1, figsize=(width, 1.4 * (model_params['n_inp'] + 1)), sharex=True)
    plt.suptitle('Prediction horizon')
    x_values_var = np.arange(run_params['num_i_c'] + run_params['pred_len']) / mtu_constant
    x_values = np.arange(run_params['num_i_c']) / mtu_constant

    for var in range(model_params['n_inp']):
        data_from = run_params['start_i_c']
        data_to = run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'] + run_params['pred_len']
        data_skip = run_params['i_c_distance']
        ax[var].plot(
            x_values_var,
            data[var, data_from:data_to:data_skip],
            'black', label='Variable: {}/{}'.format(var + 1, model_params['n_inp']))
        ax[var].set_xlim([0, x_values_var[-1]])
        ax[var].legend(loc=1)

    hor_id = model_params['n_inp']
    ax[hor_id].plot(x_values, horizons / mtu_constant, 'black', label='Horizons')
    ax[hor_id].set_xlabel('initial condition (shift from start in MTU)')
    ax[hor_id].set_ylabel('MTU')
    ax[hor_id].set_xlim([0, x_values_var[-1]])
    ax[hor_id].legend(loc=1)

    plt.savefig(os.path.join(graph_folder, 'pred_horizons.png'), dpi=300)


# Plotting colored map of error for each time-step of each initial condition

def plot_all_errors(run_params, errors, graph_folder, mtu_constant):
    Printer.print_log(True, 'Plotting colored map of all errors... ')

    errors = errors.T
    errors = np.array([errors[i, :] for i in range(errors.shape[0] - 1, 0, -1)])

    width = max(run_params['num_i_c'] / 300, 10)
    fig, ax = plt.subplots(1, figsize=(width, 10))
    plt.suptitle('Colored map of errors for each time-step and initial condition')

    ax.matshow(errors, aspect='auto', cmap=plt.cm.get_cmap('Blues'))
    ax.set_xlabel('initial condition (shift from start in MTU)')
    ax.set_ylabel('prediction time-step')
    # Flip y axis
    # Width more than one pixel

    ax.set_xticklabels([''] + list(np.arange(0, run_params['num_i_c'] + 0.1, run_params['num_i_c'] / 5) / mtu_constant))
    ax.set_yticklabels([''] + list(range(run_params['pred_len'], -1, -int(run_params['pred_len'] / 5))))

    plt.savefig(os.path.join(graph_folder, 'colored_errors.png'), dpi=300)


# Plotting prediction horizons per each variable

def plot_horizon_per_var(run_params, data, model_params, horizons, graph_folder, mtu_constant):
    Printer.print_log(True, 'Plotting prediction horizon graphs per variable... ')

    width = max(run_params['num_i_c'] / 300, 10)
    fig, ax = plt.subplots(model_params['n_inp'] * 2, figsize=(width, 2 * 1.4 * model_params['n_inp']), sharex=True)
    plt.suptitle('Prediction horizons per variable')
    x_values_var = np.arange(run_params['num_i_c'] + run_params['pred_len']) / mtu_constant
    x_values = np.arange(run_params['num_i_c']) / mtu_constant

    for var in range(model_params['n_inp']):
        data_row = var * 2
        hor_row = data_row + 1

        data_from = run_params['start_i_c']
        data_to = run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'] + run_params['pred_len']
        data_skip = run_params['i_c_distance']

        ax[data_row].plot(
            x_values_var,
            data[var, data_from:data_to:data_skip],
            'black', label='Variable: {}/{}'.format(var + 1, model_params['n_inp']))
        ax[data_row].legend(loc=1)
        ax[data_row].set_xlim([0, x_values_var[-1]])

        ax[hor_row].plot(x_values, horizons[var] / mtu_constant, 'black', label='Variable {} horizons'.format(var + 1))
        ax[hor_row].set_xlim([0, x_values_var[-1]])
        ax[hor_row].set_xlabel('initial condition (shift from start in MTU)')
        ax[hor_row].set_ylabel('MTU')
        ax[hor_row].legend(loc=1)

    plt.savefig(os.path.join(graph_folder, 'pred_horizons_per_var.png'), dpi=300)


# Plot histogram of prediction horizons

def plot_histogram(horizons, graph_folder, mtu_constant):
    Printer.print_log(True, 'Plotting prediction horizon histogram... ')

    num_bins = 30

    plt.figure()
    plt.suptitle('Prediction horizon histogram')
    plt.hist(horizons / mtu_constant, bins=num_bins, color='black')
    plt.xlim(xmin=0.0)
    plt.xlabel('MTU')
    plt.savefig(os.path.join(graph_folder, 'pred_horizons_histogram.png'), dpi=300)

    curve_points = num_bins * 50

    min_hor = np.min(horizons)
    max_hor = np.max(horizons)
    data = horizons.reshape(-1, 1)
    gmm_model = GaussianMixture(n_components=3).fit(data)
    total_curve = np.exp(gmm_model.score_samples(np.linspace(min_hor, max_hor, curve_points).reshape(-1, 1)))
    total_curve /= sum(total_curve)
    total_curve *= horizons.shape[0] / num_bins * curve_points

    plt.figure()
    plt.suptitle('Prediction horizon histogram with combined gmm')
    plt.hist(horizons / mtu_constant, bins=num_bins, color='black')
    plt.xlim(xmin=0.0)
    plt.plot(np.linspace(min_hor, max_hor, curve_points) / mtu_constant, total_curve, color='orange')
    plt.xlabel('MTU')
    plt.savefig(os.path.join(graph_folder, 'pred_horizons_histogram_combined_gmm.png'), dpi=300)

    plt.figure()
    plt.suptitle('Prediction horizon histogram with separate gmm')
    plt.hist(horizons / mtu_constant, bins=num_bins, color='black')

    for i in range(3):
        mu = gmm_model.means_[i]
        sigma = math.sqrt(gmm_model.covariances_[i])
        x = np.linspace(min_hor, max_hor, curve_points)
        curve = norm.pdf(x, mu, sigma)
        curve = curve * horizons.shape[0] / sum(curve) / num_bins * curve_points * gmm_model.weights_[i]
        plt.plot(x / mtu_constant, curve, color='orange')

    plt.xlim(xmin=0.0)
    plt.xlabel('MTU')
    plt.savefig(os.path.join(graph_folder, 'pred_horizons_histogram_separate_gmm.png'), dpi=300)

    gmm_classes = gmm_model.predict(data)
    means = [mean for mean in gmm_model.means_[:, 0]]
    class_map = np.argsort(means)
    name_map = ['hard', 'medium', 'easy']
    gmm_classes = [name_map[np.argmax(class_map == bin_class)] for bin_class in gmm_classes]
    plt.figure()
    plt.suptitle('Easy, medium and hard initial conditions')
    bin_values = [
        len([x for x in gmm_classes if x == 'hard']),
        len([x for x in gmm_classes if x == 'medium']),
        len([x for x in gmm_classes if x == 'easy'])
    ]
    # plt.hist(gmm_classes, bins=num_bins, color='black')
    plt.bar('hard', bin_values[0], color='black')
    plt.bar('medium', bin_values[1], color='black')
    plt.bar('easy', bin_values[2], color='black')
    plt.xlabel('Difficulty')
    plt.savefig(os.path.join(graph_folder, 'easy_medium_hard.png'))

    plt.figure()
    plt.suptitle('Distribution of probability')
    x = np.linspace(min_hor, max_hor, curve_points)
    p = gmm_model.predict_proba(x.reshape(-1, 1))
    p = p.cumsum(1).T

    plt.fill_between(x / mtu_constant, 0, p[0], color='gray', alpha=0.3)
    plt.fill_between(x / mtu_constant, p[0], p[1], color='gray', alpha=0.5)
    plt.fill_between(x / mtu_constant, p[1], 1, color='gray', alpha=0.7)
    plt.xlim(xmin=0.0)
    plt.ylim(0, 1)
    plt.xlabel('MTU')
    plt.ylabel('probability')
    plt.savefig(os.path.join(graph_folder, 'probabilities.png'))

    return gmm_classes


# Calculate and draw error and std plots for each histogram bin

def calc_and_draw_per_bin(horizons, errors, graph_folder, difficulty_classes, mtu_constant, threshold):
    Printer.print_log(True, 'Plotting error graphs per difficulty... ')

    fig, ax = plt.subplots(3)
    plt.suptitle('Mean error with standard deviation per difficulty')

    for i, difficulty in enumerate(['hard', 'medium', 'easy']):

        picked_horizons = [hor for i, hor in enumerate(horizons) if difficulty_classes[i] == difficulty]
        picked_errors = [error for i, error in enumerate(errors) if difficulty_classes[i] == difficulty]

        if len(picked_horizons) == 0:
            continue

        e = np.average(picked_errors, axis=0)
        std = np.std(picked_errors, axis=0)
        max_horizon = np.argmax(e > threshold)

        x_values = np.arange(e.shape[0]) / mtu_constant

        ax[i].set_title(difficulty)
        ax[i].plot(x_values, e + std, color='blue', alpha=0.4)
        ax[i].plot(x_values, e, color='red')
        ax[i].plot(x_values, e - std, color='green', alpha=0.4)
        ax[i].plot(x_values, [threshold for _ in range(e.shape[0])], 'r--', color='black')
        ax[i].set_xlim([0, max_horizon * 2 / mtu_constant])
        ax[i].set_ylim([0, threshold * 2])
        ax[i].set_ylabel('L2')

    fig.text(0.5, 0.04, 'MTU', ha='center')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.12)
    fig.savefig(os.path.join(graph_folder, 'error_std.png'), dpi=300)


# Main function starting the threads and summarizing the results

def main():

    # Loading Lorenz data

    data_set = DataSet('lorenz_96', 200, 1)
    data = data_set.data

    # Load default initialization parameters

    model_params = Model.default_model_params('esn')
    # model_params = Model.default_model_params('lstm')
    # model_params = Model.default_model_params('lstm2')
    # model_params = Model.default_model_params('ensemble_1')

    run_params = Model.default_run_params()

    model = ModelEsn(model_params, data_set.path_prefix())
    # model = ModelLstm(model_params, data_set.path_prefix())
    # model = ModelLstm2(model_params, data_set.path_prefix())
    # model = ModelEnsemble1(model_params, data_set.path_prefix())

    # Load existing prediction or load or train the model

    file_prefix = model.model_run_param_prefix_with_data(model.params_to_dict(), run_params)

    try:
        predictions = np.load(file_prefix + 'pred.npy')
        model = None
    except OSError:
        Printer.add(True, 'training the model')
        model.load_or_train(data, True)
        predictions = np.zeros((run_params['num_i_c'], model_params['n_inp'], run_params['pred_len']))
        Printer.rem(True)

    try:
        x_true_avg = np.load(file_prefix + 'x_true.npy')
    except OSError:
        Printer.print_log(True, 'Computing average Xtrue... ')
        x_true_avg = np.zeros((run_params['pred_len'],))
        for init_cond in range(
                run_params['start_i_c'],
                run_params['start_i_c'] + run_params['num_i_c'] * run_params['i_c_distance'],
                run_params['i_c_distance']):
            for dt in range(run_params['pred_len']):
                x_true_avg[dt] += np.linalg.norm(data[:, init_cond + dt])
        x_true_avg /= run_params['num_i_c']
        try:
            np.save(file_prefix + 'x_true.npy', x_true_avg)
        except OSError:
            pass
        Printer.print_log(True, 'Computing average Xtrue... Done')

    # Run threads

    thread_count = 2
    threads = []

    horizons = np.zeros((model_params['n_inp'] + 1, run_params['num_i_c'], ))
    errors = np.zeros((run_params['num_i_c'], run_params['pred_len']))

    for i in range(thread_count):
        thread = threading.Thread(target=thread_main, args=(
            model_params, data, run_params, horizons, i, thread_count, model, x_true_avg, predictions, errors
        ))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    # Save predictions if not already saved

    try:
        np.save(file_prefix + 'pred.npy', np.array(predictions))
    except OSError:
        pass

    # Generate cumulative prediction horizon plot

    Printer.clear(True)
    np.random.seed()
    graph_folder = os.path.join(data_set.path_prefix(), 'pred_horizon_') + str(np.random.random())[2:]
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)
    with open(os.path.join(graph_folder, 'model_params.json'), 'w') as out_file:
        json.dump(model_params, out_file)
    with open(os.path.join(graph_folder, 'default_run.json'), 'w') as out_file:
        json.dump(run_params, out_file)

    threshold = model_params['n_inp'] * 0.0375

    plot_horizon(run_params, data, model_params, horizons[0], graph_folder, data_set.dt_per_mtu)
    print()

    # Generate colored matrix

    plot_all_errors(run_params, errors, graph_folder, data_set.dt_per_mtu)
    print()

    # Generate per variable prediction horizon plot

    plot_horizon_per_var(run_params, data, model_params, horizons[1:], graph_folder, data_set.dt_per_mtu)
    print()

    # Generate histogram

    difficulty_classes = plot_histogram(horizons[0], graph_folder, data_set.dt_per_mtu)
    print()

    # Generate histogram bin error graphs

    calc_and_draw_per_bin(horizons[0], errors, graph_folder, difficulty_classes, data_set.dt_per_mtu, threshold)
    print()


if __name__ == '__main__':
    main()
