import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.models import load_model

from script.model import Model, DataSet
from script.model_lstm import ModelLstm


def main():

    # Floating point precisions
    fp_precision = [
        ("float64", np.double, "64 bit"),
        ("float32", np.single, "32 bit"),
        ("float16", np.half, "16 bit")
    ]

    # Loading Lorenz data

    data_set = DataSet("lorenz_96", 200, 1)
    data = data_set.data
    start_test_ic = 500000
    prediction_len = 500

    # Load default initialization parameters

    model_params = Model.default_model_params("lstm")

    # Do predictions with different precision
    # offsets = np.array([0, 5000, 25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000])
    # offsets = np.concatenate((offsets, offsets + 1000, offsets + 2000, offsets + 3000))
    offsets = np.arange(1000, 1000 + 20 * 1000, 1000)
    assert offsets.shape[0] == 20
    errors = np.zeros((3, len(offsets), prediction_len))
    for i, (fp_pr, np_pr, _) in enumerate(fp_precision):

        print("precision: " + fp_pr)

        tf.keras.backend.set_floatx(fp_pr)

        # Convert data
        con_data = np.copy(data).astype(np_pr)

        # Prepare model
        model = ModelLstm(model_params, data_set.path_prefix())
        try:
            # Load model
            model.model = load_model(fp_pr + "_lstm_model.h5")
            model.model._make_predict_function()
            print("loaded")
        except OSError:
            # Train model
            model.train(con_data, True)
            # Save model explicitly (changing precision interferes with the rest of the stack)
            model.model.save(fp_pr + "_lstm_model.h5")
            print("trained")

        # Loop over offsets

        for j, offset in enumerate(offsets):

            run_params = {
                "warm_up_size": 3,
                "num_i_c": 1,
                "pred_len": prediction_len,
                "start_i_c": start_test_ic + offset,
                "i_c_distance": 1
            }

            # Calculate x_true
            x_true_avg = np.zeros((run_params["pred_len"],))
            for init_cond in range(
                    run_params["start_i_c"],
                    run_params["start_i_c"] + run_params["num_i_c"] * run_params["i_c_distance"],
                    run_params["i_c_distance"]):
                for dt in range(run_params["pred_len"]):
                    x_true_avg[dt] += np.linalg.norm(con_data[:, init_cond + dt])
            x_true_avg /= run_params["num_i_c"]

            # Predict
            warm_up_input = con_data[:, run_params["start_i_c"] - run_params["warm_up_size"]:run_params["start_i_c"]]
            prediction = model.predict(warm_up_input, run_params, 1, False)

            # Calculate error
            comp_data = con_data[:, run_params["start_i_c"]:run_params["start_i_c"] + run_params["pred_len"]]
            errors[i, j] = np.linalg.norm(prediction - comp_data, axis=0) / x_true_avg

    np.save("LSTM_precision.npy", errors)
    print(errors[0, 0, 200:210])
    print(errors[1, 0, 200:210])
    print(errors[2, 0, 200:210])

    threshold = 0.3

    # Calculate horizons
    horizons = np.zeros((3, len(offsets)))
    for i, one_precision in enumerate(errors):
        for j, one_offset in enumerate(one_precision):
            horizons[i, j] = np.argmax(one_offset > threshold)

    # Plot the graph

    line_width = 0.75
    colors = ["C0", "C1", "C2"]

    for i, error in enumerate(errors[:, 0, :]):
        plt.plot(error, color=colors[i])

    for i, hor in enumerate(horizons[:, 0]):
        plt.plot([hor, hor], [0, threshold], color=colors[i], linewidth=line_width)

    plt.plot([threshold for _ in range(prediction_len)], color="black", linewidth=line_width)

    plt.legend([precision[2] for precision in fp_precision])
    plt.xlabel("Testing Step")
    plt.ylabel("Relative (Norm) Error")
    plt.title("LSTM")
    plt.xlim([0, prediction_len])
    plt.ylim([0, 1.6])
    plt.savefig("LSTM precision.png")

    # Calculate table

    print(horizons)

    for i in range(len(fp_precision)):
        print(
            fp_precision[i][2] + ": " +
            "%.2e" % (np.mean(horizons[i]) / 200.0) + " ± " +
            "%.2e" % (np.std(horizons[i]) / 200.0)
        )


if __name__ == "__main__":
    main()
