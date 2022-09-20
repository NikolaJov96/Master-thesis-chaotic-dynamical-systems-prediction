from script.model import Model, DataSet

import json
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image


def get_file_name(graph_folder, ic):

    file_name = 'color_map_{}.png'.format(ic)
    return os.path.join(graph_folder, file_name)


def main():

    data_set = DataSet('lorenz_96', 200, 1)

    run_params = Model.default_run_params()

    model_name = 'esn'
    model_params = Model.default_model_params(model_name)

    prefix = os.path.join(data_set.path_prefix(), Model.model_run_params_prefix(model_params, run_params))
    prediction = np.load(prefix + 'pred.npy')

    graph_folder = os.path.join(data_set.path_prefix(), 'color_map_') + str(np.random.random())[2:]

    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)
    with open(os.path.join(graph_folder, '{}_params.json'.format(model_name)), 'w') as out_file:
        json.dump(model_params, out_file)
    with open(os.path.join(graph_folder, 'run_params.json'), 'w') as out_file:
        json.dump(run_params, out_file)

    initial_condition = 0
    num_initial_conditions = 10
    step = 5

    # Draw images

    for i in range(num_initial_conditions):

        ic = initial_condition + i * step

        print('\r{}/{}'.format(i + 1, num_initial_conditions), end='')

        original_data = data_set.data[
                        :,
                        run_params['start_i_c'] + ic:
                        run_params['start_i_c'] + ic + run_params['pred_len']
        ]

        predicted_data = prediction[ic]

        difference = predicted_data - original_data

        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('Color mapped prediction {}/{}'.format(i + 1, num_initial_conditions))
        fig.tight_layout()
        color_map = plt.get_cmap('jet')

        def draw_graph(ax, data, title):
            img0 = ax.imshow(data, interpolation='nearest', aspect='auto', cmap=color_map,
                             origin='lower', vmin=-3, vmax=3)
            ax.get_yaxis().set_visible(False)
            ax.set_title(title)
            fig.colorbar(img0, cmap=color_map, ax=ax)

        draw_graph(axs[0], original_data, 'Original data')
        draw_graph(axs[1], predicted_data, 'Prediction data')
        draw_graph(axs[2], difference, 'Difference')

        plt.subplots_adjust(top=0.9)

        plt.savefig(get_file_name(graph_folder, ic))

    # Produce GIF

    images = []

    for i in range(num_initial_conditions):

        ic = initial_condition + i * step
        images.append(Image.open(get_file_name(graph_folder, ic)))

    images[0].save(os.path.join(graph_folder, 'animation.gif'), save_all=True,
                   append_images=images[1:], optimize=False, duration=50, loop=0)


if __name__ == '__main__':
    main()
