# Generate statistics about impact of Win and A seeds on prediction horizon

from script.model import Model, DataSet
from script.model_esn import ModelEsn
from script.printer import Printer

import math
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import threading
import time


# Calculate relative error

def calc_error(data, res_params, run_params, verbose, model):

    initial_conditions = range(
        run_params['start_i_c'],
        run_params['start_i_c'] +
        run_params['num_i_c'] * run_params['i_c_distance'],
        run_params['i_c_distance']
    )

    file_prefix = model.model_run_params_prefix(res_params, run_params)

    try:
        Printer.print_log(verbose, 'Loading predictions and relative error... ')
        predictions = np.load(file_prefix + 'pred.npy')
        e = np.load(file_prefix + 'e.npy')
        std = np.load(file_prefix + 'std.npy')
        Printer.print_log(verbose, 'Loading relative error... Done')
    except FileNotFoundError:
        Printer.add(verbose, 'Calculating error')
        Printer.print_log(verbose, 'Computing average Xtrue... ')
        x_true_avg = np.zeros((run_params['pred_len'], ))
        for init_cond in initial_conditions:
            for dt in range(run_params['pred_len']):
                x_true_avg[dt] += np.linalg.norm(data[:, init_cond + dt])
        x_true_avg /= run_params['num_i_c']
        Printer.print_log(verbose, 'Computing average Xtrue... Done')

        e = np.zeros((run_params['pred_len'], ))
        std = np.zeros((run_params['pred_len'], ))
        predictions = np.zeros((len(initial_conditions), res_params['n_inp'], run_params['pred_len']))

        start_time = time.time()
        for i, init_cond in enumerate(initial_conditions):
            Printer.add(verbose, 'IC: {}/{} - {}s'.
                        format(i + 1, run_params['num_i_c'], int(time.time() - start_time)))
            Printer.print_log(verbose)
            warm_up_input = data[:, init_cond - run_params['warm_up_size']:init_cond]

            prediction = model.predict(warm_up_input, run_params, i, verbose)
            predictions[i, :] = prediction
            e += np.linalg.norm(prediction - data[:, init_cond:init_cond + run_params['pred_len']],
                                axis=0
                                ) / x_true_avg
            std += np.std(prediction - data[:, init_cond:init_cond + run_params['pred_len']], axis=0)

            Printer.rem(verbose)

        e /= run_params['num_i_c']
        std /= run_params['num_i_c']

        Printer.print_log(verbose, 'Saving predictions, error and std... ')
        np.save(file_prefix + 'pred.npy', predictions)
        np.save(file_prefix + 'e.npy', e)
        np.save(file_prefix + 'std.npy', std)
        Printer.print_log(verbose, 'Saving error and std... Done')

        Printer.rem(verbose)

    return predictions, e, std


# Min thread function

def thread_main(seed_combinations, res_params, num_win_seeds, num_a_seeds,
                data_set, run_params, e_and_std, ind, threads):

    data = data_set.data
    verbose = ind == 0
    trained_models = []
    for count, (i, j, win_seed, a_seed) in enumerate(seed_combinations):
        res_params_new = dict(res_params)
        res_params_new['win_seed'] = win_seed
        res_params_new['a_seed'] = a_seed

        Printer.clear(verbose)
        Printer.add(verbose, 'Thread: {}/{}'.format(ind + 1, threads))
        Printer.add(verbose, 'Win seed: {}'.format(win_seed, i + 1, num_win_seeds))
        Printer.add(verbose, 'A seed: {}'.format(a_seed, j + 1, num_a_seeds))
        Printer.add(verbose, '{}/{}'.format(count + 1, len(seed_combinations)))

        Printer.print_log(verbose)
        model = ModelEsn(res_params_new, data_set.path_prefix())
        model.load_or_train(data, verbose)
        trained_models.append((model, i, j, win_seed, a_seed))

    for count, (model, i, j, win_seed, a_seed) in enumerate(trained_models):
        res_params_new = dict(res_params)
        res_params_new['a_seed'] = a_seed
        res_params_new['win_seed'] = win_seed

        Printer.clear(verbose)
        Printer.add(verbose, 'Thread: {}/{}'.format(ind + 1, threads))
        Printer.add(verbose, 'Win seed: {}'.format(win_seed, i + 1, num_win_seeds))
        Printer.add(verbose, 'A seed: {}'.format(a_seed, j + 1, num_a_seeds))
        Printer.add(verbose, '{}/{}'.format(count + 1, len(seed_combinations)))

        predictions, e, std = calc_error(data, res_params_new, run_params, verbose, model)
        e_and_std[i][j] = (win_seed, a_seed, predictions, e, std)


# Plot Win and A grid of graphs

def plot_w_in_a_grid(num_win_seeds, num_a_seeds, res_params, run_params, e_and_std,
                     graph_folder, mtu_constant, threshold):

    Printer.print_log(True, 'Plotting Win and A grid of graphs... ')
    fig_size_x = num_win_seeds * 5
    fig_size_y = num_a_seeds * 3
    fig, ax = plt.subplots(num_win_seeds, num_a_seeds, figsize=(fig_size_x, fig_size_y))
    plt.suptitle('ESN Win and A seeds')
    max_e_seen = 0
    x_values = np.arange(run_params['pred_len']) / mtu_constant

    for i in range(num_win_seeds):
        for j in range(num_a_seeds):
            win_seed = e_and_std[i][j][0]
            a_seed = e_and_std[i][j][1]
            # predictions = e_and_std[i][j][2]
            e = e_and_std[i][j][3]
            std = e_and_std[i][j][4]

            ax[i, j].plot(x_values, e + std, color='blue')
            ax[i, j].plot(x_values, e, color='red')
            ax[i, j].plot(x_values, e - std, color='green')
            ax[i, j].plot(x_values, [threshold for _ in range(run_params['pred_len'])], 'r--', color='black')
            max_e_seen = max(max_e_seen, max(e + std))
            ax[i, j].set_ylim([0, max_e_seen * 1.1])
            ax[i, j].set_xlim([0, run_params['pred_len'] / mtu_constant])
            pred_horizon = x_values[np.argmax(e > threshold)]
            ax[i, j].axvline(x=pred_horizon, color='black', linestyle='--',
                             label='Pred horizon: {}'.format(pred_horizon))
            ax[i, j].legend(loc='upper right')
            if i == num_win_seeds - 1:
                ax[i, j].set_xlabel('A seed: {}'.format(a_seed))
            if j == 0:
                ax[i, j].set_ylabel('Win seed: {}'.format(win_seed))

    plt.xticks(np.arange(0, run_params['pred_len'] / mtu_constant + 0.1, 0.5))
    fig.text(0.5, 0.04, 'MTU', ha='center')
    fig.text(0.04, 0.5, 'L2(t)', va='center', rotation='vertical')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

    # Saving

    Printer.print_log(True, 'Saving graph... ')
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)
    plt.savefig(os.path.join(graph_folder, 'grid_graph.png'), dpi=300)
    with open(os.path.join(graph_folder, 'res_params.json'), 'w') as out_file:
        json.dump(res_params, out_file)
    with open(os.path.join(graph_folder, 'default_run.json'), 'w') as out_file:
        json.dump(run_params, out_file)
    Printer.print_log(True, 'Saving graph...  Done')


# Plot individual error graph

def plot_individual(es, title, run_params, file_name, graph_folder, mtu_constant, threshold):

    e_avg = np.average(es, axis=0)
    e_std = np.std(es, axis=0)

    Printer.print_log(True, 'Plotting {} graph... '.format(title))

    fig, ax = plt.subplots(1)
    plt.suptitle(title)
    max_e_seen = 0
    x_values = np.arange(run_params['pred_len']) / mtu_constant

    ax.plot(x_values, e_avg + e_std, color='blue')
    ax.plot(x_values, e_avg, color='red')
    ax.plot(x_values, e_avg - e_std, color='green')
    ax.plot(x_values, [threshold for _ in range(run_params['pred_len'])], 'r--', color='black')
    max_e_seen = max(max_e_seen, max(e_avg + e_std))
    ax.set_ylim([0, max_e_seen * 1.1])
    ax.set_xlim([0, run_params['pred_len'] / mtu_constant])
    ax.axvline(x=x_values[np.argmax(e_avg > threshold)], color='black', linestyle='--')
    ax.set_xlabel('MTU')
    ax.set_ylabel('L2(t)')
    plt.xticks(np.arange(0, run_params['pred_len'] / mtu_constant + 0.1, 0.5))

    # Saving

    Printer.print_log(True, 'Saving graph... ')
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)
    plt.savefig(os.path.join(graph_folder, file_name + '.png'), dpi=300)
    Printer.print_log(True, 'Saving graph...  Done')


# Main function starting the threads and summarizing the results

def main():

    # Loading Lorenz data

    data_set = DataSet('lorenz_96', 200, 1)

    # Load default initialization parameters

    res_params = Model.default_model_params('esn')
    run_params = Model.default_run_params()

    # Iterate over Win and A seeds

    win_seeds = [100, 101, 102, 103]
    num_win_seeds = len(win_seeds)
    a_seeds = [200, 201, 202, 203]
    num_a_seeds = len(a_seeds)
    seed_combinations = [[i, j, x, y] for i, x in enumerate(win_seeds) for j, y in enumerate(a_seeds)]
    num_combinations = len(seed_combinations)

    thread_count = 2
    tasks_per_thread = math.ceil(num_combinations / thread_count)
    threads = []

    e_and_std = [[None for _ in range(num_win_seeds)] for _ in range(num_a_seeds)]

    for i in range(thread_count):
        thread = threading.Thread(target=thread_main, args=(
            seed_combinations[i * tasks_per_thread:min((i + 1) * tasks_per_thread, num_combinations)],
            res_params, num_win_seeds, num_a_seeds, data_set, run_params, e_and_std, i, thread_count
        ))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print()

    Printer.clear(True)
    np.random.seed()
    graph_folder = os.path.join(data_set.path_prefix(), 'Win_A_graph_') + str(np.random.random())[2:]
    threshold = res_params['n_inp'] * 0.0375

    # Plotting Win and A grid graph

    plot_w_in_a_grid(num_win_seeds, num_a_seeds, res_params, run_params, e_and_std,
                     graph_folder, data_set.dt_per_mtu, threshold)

    # Plotting Average graphs per Win A seeds

    e_per_w_in = np.zeros((num_win_seeds, run_params['pred_len']))
    e_per_a = np.zeros((num_a_seeds, run_params['pred_len']))
    e_all = np.zeros((num_win_seeds * num_a_seeds, run_params['pred_len']))
    for i in range(num_win_seeds):
        for j in range(num_a_seeds):
            e_per_w_in[i, :] += e_and_std[i][j][3]
            e_per_a[j, :] += e_and_std[i][j][3]
            e_all[i * num_a_seeds + j, :] = e_and_std[i][j][3]
    e_per_w_in /= num_a_seeds
    e_per_a /= num_win_seeds

    plot_individual(e_per_w_in, 'ESN variation per Win seed', run_params, 'per_w_in_seed_graph.png',
                    graph_folder, data_set.dt_per_mtu, threshold)
    plot_individual(e_per_a, 'ESN variation per A seed', run_params, 'per_a_seed_graph.png',
                    graph_folder, data_set.dt_per_mtu, threshold)
    plot_individual(e_all, 'ESN variation per Win and A seeds', run_params, 'per_w_in_and_a_seeds_graph.png',
                    graph_folder, data_set.dt_per_mtu, threshold)


if __name__ == '__main__':
    main()
