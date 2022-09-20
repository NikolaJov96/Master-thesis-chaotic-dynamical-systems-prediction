import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ESN_Paper_Code_Table_min import get_fourier


def main1():

    dir_path = [
        "D2R2_exponents/",
        "ESN_exponents/",
        "ESN_100/",
        "D2R2_100/",
        "D2R2_mantissas_smaller/",
        "D2R2_all_mantissas/",
    ][3]

    if dir_path == "D2R2_exponents/":
        plot64 = False
        plot32 = False
        plot16 = False
        draw_mantissa = [12]
        # draw_exponents = [9, 8, 7, 6, 5, 4, 3]
        draw_exponents = [6, 5, 4]
    elif dir_path == "ESN_exponents/":
        plot64 = False
        plot32 = False
        plot16 = False
        draw_mantissa = [12]
        draw_exponents = [9, 8, 7, 6, 5, 4, 3]
        # draw_exponents = [6, 5, 4]
    elif dir_path == "ESN_100/":
        plot64 = True
        plot32 = True
        plot16 = True
        # draw_mantissa = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
        draw_mantissa = [12, 9, 4]
        draw_exponents = [10]
    elif dir_path == "D2R2_100/":
        plot64 = True
        plot32 = True
        plot16 = True
        # draw_mantissa = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
        draw_mantissa = [12, 10, 9, 8, 4]
        draw_exponents = [10]
    elif dir_path == "D2R2_mantissas_smaller/":
        plot64 = True
        plot32 = True
        plot16 = True
        # draw_mantissa = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
        draw_mantissa = [9, 8, 4]
        draw_exponents = [10]
    elif dir_path == "D2R2_all_mantissas/":
        plot64 = True
        plot32 = True
        plot16 = True
        draw_mantissa = [32, 16, 15, 14, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        # draw_mantissa = [9, 8, 4]
        draw_exponents = [10]

    tabledat = np.load(dir_path + "64bit.npy")
    tabledat32 = np.load(dir_path + "32bit.npy")
    tabledat16 = np.load(dir_path + "16bit.npy")
    table_dat = np.load(dir_path + "red_bit.npy")
    all_preds = np.load(dir_path + "all_preds.npy")
    pplot = np.load(dir_path + "plot64.npy")
    pplot32 = np.load(dir_path + "plot32.npy")
    pplot16 = np.load(dir_path + "plot16.npy")
    plots = np.load(dir_path + "plot_red.npy")
    mantissas = np.load(dir_path + "mantissas.npy")
    exponents = np.load(dir_path + "exponents.npy")

    print(tabledat)
    print(tabledat32)
    print(tabledat16)
    print(table_dat)
    print("64 bit: %.2e" % (np.mean(tabledat) / 200.0) + " ± " + "%.2e" % (np.std(tabledat) / 200.0))
    print("32 bit: %.2e" % (np.mean(tabledat32) / 200.0) + " ± " + "%.2e" % (np.std(tabledat32) / 200.0))
    print("16 bit: %.2e" % (np.mean(tabledat16) / 200.0) + " ± " + "%.2e" % (np.std(tabledat16) / 200.0))

    means = np.mean(table_dat, 2)
    stds = np.std(table_dat, 2)
    for i in range(len(mantissas)):
        for j in range(len(exponents)):
            print("%d bit mantissa, %d bit exponent: %.2e ± %.2e" % (mantissas[i], exponents[j], means[i, j] / 200.0, stds[i, j] / 200.0))


    print("asd", all_preds.shape)
    recalc_mean = np.average(all_preds, 2)
    recalc_std = np.std(all_preds, 2)
    print(recalc_mean.shape)
    for i in range(len(mantissas)):
        for j in range(len(exponents)):
            ind = np.argmax(recalc_mean[i][j] > 0.3)
            print("%d bit mantissa, %d bit exponent: %.2e MTU, std: %.2e" % (
            mantissas[i], exponents[j], ind / 200.0, recalc_std[i, j, ind] / 200.0))


    # fig = plt.figure()
    color_id = 0
    if plot64:
        plt.plot(pplot, label="64 bit", color="C%d" % color_id)
        horizon = np.argmax(pplot > 0.3)
        plt.plot([horizon, horizon], [0, 0.3], color="C%d" % color_id, linewidth=0.75)
        color_id += 1
    if plot32:
        plt.plot(pplot32, label="32 bit", color="C%d" % color_id)
        horizon = np.argmax(pplot32 > 0.3)
        plt.plot([horizon, horizon], [0, 0.3], color="C%d" % color_id, linewidth=0.75)
        color_id += 1
    if plot16:
        plt.plot(pplot16, label="16 bit", color="C%d" % color_id)
        horizon = np.argmax(pplot16 > 0.3)
        plt.plot([horizon, horizon], [0, 0.3], color="C%d" % color_id, linewidth=0.75)
        color_id += 1
    for i in range(len(mantissas)):
        if mantissas[i] in draw_mantissa:
            for j in range(len(exponents)):
                if exponents[j] in draw_exponents:
                    plt.plot(plots[i, j], label=str(mantissas[i]) + "m, " + str(exponents[j]) + "e", color="C%d" % color_id)
                    try:
                        horizon = np.argmax(plots[i, j] > 0.3)
                    except Exception:
                        horizon = 0.0
                    plt.plot([horizon, horizon], [0, 0.3], color="C%d" % color_id, linewidth=0.75)
                    color_id += 1
    plt.plot([0.3 for _ in range(500)], color="black", linewidth=0.75)
    plt.legend(loc='upper right')
    plt.xlabel('Testing Step')
    plt.ylabel('Relative (Norm) Error')
    plt.title('Lorenz96 for varying precision')
    plt.xlim([0, 500])
    plt.ylim([0, 1.6])
    plt.savefig(dir_path + "precision_edited.png")


def main2_plt(data, mantissas, filename, name, x_name):
    plt.figure()
    plt.xlabel(x_name + ' size (bits)')
    plt.ylabel('Horizon with standard deviation (MTU)')
    plt.title(name + ' varying precision on Lorenz96')
    plt.errorbar(range(len(mantissas), 0, -1), data[:, 0], data[:, 1], fmt='-o')
    plt.xticks(range(len(mantissas), 0, -1), mantissas)
    plt.savefig(filename + ".png")


def main2():
    mantissas = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    d2r2_data = np.array([
        np.array([1.83e+00, 6.55e-01]),
        np.array([1.65e+00, 6.88e-01]),
        np.array([1.47e+00, 5.99e-01]),
        np.array([1.47e+00, 5.89e-01]),
        np.array([6.84e-01, 1.37e-01]),
        np.array([4.39e-01, 7.14e-02]),
        np.array([2.31e-01, 3.29e-02]),
        np.array([2.26e-01, 3.44e-02]),
        np.array([6.75e-02, 2.50e-03]),
        np.array([2.55e-02, 1.50e-03])
    ])
    esn_data = np.array([
        np.array([1.13e+00, 2.05e-01]),
        np.array([1.07e+00, 4.70e-01]),
        np.array([6.30e-01, 1.51e-01]),
        np.array([2.99e-01, 8.42e-02]),
        np.array([2.06e-01, 9.22e-02]),
        np.array([1.13e-01, 4.34e-02]),
        np.array([5.95e-02, 2.16e-02]),
        np.array([5.25e-02, 2.41e-02]),
        np.array([2.10e-02, 5.39e-03]),
        np.array([1.05e-02, 2.69e-03])
    ])
    d2r2_exp_data = np.array([
        np.array([1.63e+00, 4.78e-01]),
        np.array([1.63e+00, 4.78e-01]),
        np.array([1.63e+00, 4.78e-01]),
        np.array([1.63e+00, 4.78e-01]),
        np.array([1.52e+00, 4.47e-01]),
        np.array([2.90e-02, 1.74e-02]),
        np.array([0.00e+00, 0.00e+00])
    ])
    esn_exp_data = np.array([
        np.array([1.24e+00, 3.92e-01]),
        np.array([1.24e+00, 3.92e-01]),
        np.array([1.24e+00, 3.92e-01]),
        np.array([1.24e+00, 3.92e-01]),
        np.array([1.24e+00, 3.92e-01]),
        np.array([2.55e-01, 1.26e-01]),
        np.array([6.00e-03, 2.00e-03])
    ])

    main2_plt(d2r2_data, mantissas, "d2r2_mean_std", "D2R2", "Mantissa")
    main2_plt(esn_data, mantissas, "esn_mean_std", "ESN", "Mantissa")
    main2_plt(d2r2_exp_data, range(9, 2, -1), "d2r2_exp_mean_std", "D2R2", "Exponent")
    main2_plt(esn_exp_data, range(9, 2, -1), "esn_exp_mean_std", "ESN", "Exponent")

    plt.figure()
    plt.xlabel('Mantissa size (bits)')
    plt.ylabel('Horizon with standard deviation (MTU)')
    plt.title('Varying precision on Lorenz96')
    plt.errorbar(mantissas, d2r2_data[:, 0], d2r2_data[:, 1], fmt='-o', label="D2R2")
    plt.errorbar(mantissas, esn_data[:, 0], esn_data[:, 1], fmt='-o', label="ESN")
    plt.xticks(mantissas)
    plt.savefig("combined_mean_std.png")

    plt.figure()
    plt.xlabel('Exponent size (bits)')
    plt.ylabel('Horizon with standard deviation (MTU)')
    plt.title('Varying precision on Lorenz96')
    plt.errorbar(range(9, 2, -1), d2r2_exp_data[:, 0], d2r2_exp_data[:, 1], fmt='-o', label="D2R2")
    plt.errorbar(range(9, 2, -1), esn_exp_data[:, 0], esn_exp_data[:, 1], fmt='-o', label="ESN")
    plt.legend()
    plt.xticks(range(9, 2, -1))
    plt.savefig("combined_exp_mean_std.png")

    d2r2_data_all = np.array([
        np.array([2.00e+00, 6.53e-01]),
        np.array([2.00e+00, 6.56e-01]),
        np.array([1.99e+00, 6.58e-01]),
        np.array([2.03e+00, 7.29e-01]),
        np.array([2.18e+00, 9.53e-01]),
        np.array([1.83e+00, 6.55e-01]),
        np.array([1.65e+00, 6.88e-01]),
        np.array([1.47e+00, 5.99e-01]),
        np.array([1.47e+00, 5.89e-01]),
        np.array([6.84e-01, 1.37e-01]),
        np.array([4.39e-01, 7.14e-02]),
        np.array([2.31e-01, 3.29e-02]),
        np.array([2.26e-01, 3.44e-02]),
        np.array([6.75e-02, 2.50e-03]),
        np.array([2.55e-02, 1.50e-03]),
        np.array([8.50e-03, 2.29e-03])
    ])
    mantissas = [32, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    main2_plt(d2r2_data_all, mantissas, "d2r2_all_mean_std", "D2R2", "Mantissa")


def main3_plot(lstm_data, d2r2_data, esn_data, filename, name):
    plt.figure()
    plt.plot(lstm_data, label="LSTM", color="C0")
    plt.plot(d2r2_data, label="D2R2", color="C1")
    plt.plot(esn_data, label="ESN", color="C2")

    hors = [
        np.argmax(lstm_data > 0.3),
        np.argmax(d2r2_data > 0.3),
        np.argmax(esn_data > 0.3)
    ]
    for i in range(3):
        plt.plot([hors[i], hors[i]], [0, 0.3], color="C%d" % i, linewidth=0.75)

    plt.plot([0.3 for _ in range(500)], color="black", linewidth=0.75)

    plt.legend(loc='upper right')
    plt.xlabel('Testing Step')
    plt.ylabel('Relative (Norm) Error')
    plt.title('Lorenz96 prediction example %s bit' % name)
    plt.xlim([0, 500])
    plt.ylim([0, 1.6])
    plt.savefig(filename + ".png")


def main3():
    lstm_data = np.load("LSTM_precision.npy")
    d2r2_data_32 = np.load("D2R2_mantissas/plot32.npy")
    d2r2_data_16 = np.load("D2R2_mantissas/plot16.npy")
    esn_data_32 = np.load("ESN/plot32.npy")
    esn_data_16 = np.load("ESN/plot16.npy")

    main3_plot(lstm_data[1], d2r2_data_32, esn_data_32, "32_bit", "32")
    main3_plot(lstm_data[2], d2r2_data_16, esn_data_16, "16_bit", "16")


def main4():
    d2r2_data = np.array([
        # np.array([2.00e+00, 6.53e-01]),  # 32 IEEE
        # np.array([1.86e+00, 7.42e-01]),  # 16 IEEE
        np.array([1.47e+00, 5.99e-01]),  # 10 mantissa
        np.array([1.34e+00, 4.29e-01]),
        np.array([1.34e+00, 4.47e-01]),
        np.array([7.35e-01, 2.44e-01]),
        np.array([4.46e-01, 1.30e-01]),
        np.array([2.26e-01, 3.32e-02]),
        np.array([2.20e-01, 3.24e-02]),
        np.array([6.83e-02, 2.93e-03]),
        # np.array([8.50e-03, 2.29e-03]),  # 2 mantissa
    ])
    esn_data = np.array([
        # np.array([1.25e+00, 4.40e-01]),  # 32 IEEE
        # np.array([4.17e-01, 1.68e-01]),  # 16 IEEE
        np.array([6.02e-01, 2.60e-01]),  # 10 mantissa
        np.array([3.91e-01, 1.52e-01]),
        np.array([1.76e-01, 7.66e-02]),
        np.array([9.61e-02, 5.22e-02]),
        np.array([6.60e-02, 1.86e-02]),
        np.array([4.24e-02, 1.52e-02]),
        np.array([2.10e-02, 6.72e-03]),
        np.array([8.25e-03, 4.32e-03]),
        # np.array([8.50e-03, 2.29e-03]),  # 2 mantissa, to update
    ])

    plt.figure()
    plt.xlabel('Mantissa size (bits)')
    plt.ylabel('Horizon with standard deviation (MTU)')
    # plt.title('Varying precision on Lorenz96')
    # plt.errorbar(range(8, 0, -1), d2r2_data[:, 0], d2r2_data[:, 1], fmt='-o', label="D2R2")
    # plt.errorbar(range(8, 0, -1), esn_data[:, 0], esn_data[:, 1], fmt='-o', label="ESN")
    plt.plot(range(8, 0, -1), d2r2_data[:, 0], label="D2R2", color="C0", marker="o")
    plt.plot(range(8, 0, -1), esn_data[:, 0], label="ESN", color="C1", marker="o")
    plt.legend(loc='upper left')
    plt.xticks(range(8, 0, -1), list(range(10, 2, -1)))
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig("combined_ieee_mantissa.png")




    ratio = d2r2_data[0, 0] / esn_data[0, 0]

    d2r2_data = d2r2_data / d2r2_data[0, 0]
    fig, ax1 = plt.subplots(figsize=[7.0, 4.8])
    ax1.set_xlabel('Mantissa size (bits)')
    ax1.set_ylabel('Normalized prediction horizon with the value\nusing a $10$ bit mantissa as a baseline for D2R2')
    # ax1.errorbar(range(10, 2, -1), d2r2_data[:, 0], d2r2_data[:, 1], fmt='-o', label="D2R2", color="C0")
    ax1.plot(range(10, 2, -1), d2r2_data[:, 0], label="D2R2", color="C0", marker="o")
    ax1.legend(loc='upper left')
    # ax1.set_xticks(range(8, 0, -1), list(range(10, 2, -1)))
    ax1.set_xticks(list(range(10, 2, -1)))
    ax1.set_yticks(np.arange(0.0, 1.2, 0.2))
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.grid(linestyle='--')
    ax1.set_ylim((0.0, 1.2))

    esn_data = esn_data / esn_data[0, 0]
    ax2 = ax1.twinx()
    ax2.set_ylabel('Normalized prediction horizon with the value\nusing a $10$ bit mantissa as a baseline for ESN')
    # ax2.errorbar(range(10, 2, -1), esn_data[:, 0], esn_data[:, 1], fmt='-o', label="ESN", color="C1")
    ax2.plot(range(10, 2, -1), esn_data[:, 0], label="ESN", color="C1", marker="o")
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.legend(loc='lower right')
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylim((0.0, 1.2 * ratio))

    plt.tight_layout()
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.95, bottom=0.1)
    plt.savefig("combined_ieee_mantissa_scaled.png")


    # d2r2_data = d2r2_data / d2r2_data[0, 0] * 100
    # esn_data = esn_data / esn_data[0, 0] * 100

    # plt.figure()
    # plt.xlabel('Mantissa size (bits)')
    # plt.ylabel('Horizon with standard deviation (%)')
    # # plt.title('D2R2 varying precision in relation to 32 bit IEEE float')
    # plt.errorbar(range(8, 0, -1), d2r2_data[:, 0], d2r2_data[:, 1], fmt='-o')
    # plt.xticks(range(8, 0, -1), list(range(10, 1, -1)))
    # plt.grid(linestyle='--')
    # plt.tight_layout()
    # plt.savefig("d2r2_ieee_mantissa.png")
    #
    # plt.figure()
    # plt.xlabel('Mantissa size (bits)')
    # plt.ylabel('Horizon with standard deviation (%)')
    # # plt.title('ESN varying precision in relation to 32 bit IEEE float')
    # plt.errorbar(range(8, 0, -1), esn_data[:, 0], esn_data[:, 1], fmt='-o')
    # plt.xticks(range(8, 0, -1), list(range(10, 1, -1)))
    # plt.grid(linestyle='--')
    # plt.tight_layout()
    # plt.savefig("esn_ieee_mantissa.png")


def main5():

    lstm_data = np.load("LSTM_precision.npy")
    print(lstm_data.shape)
    lstm_data_32 = lstm_data[1]
    lstm_plot_32 = np.mean(lstm_data_32, 0)
    lstm_std_32 = np.std(lstm_data_32, 0)
    lstm_data_16 = lstm_data[2]
    lstm_plot_16 = np.mean(lstm_data_16, 0)[:500]
    lstm_std_16 = np.std(lstm_data_16, 0)[:500]

    bits = 12

    esn_data = np.load("esn_100/all_preds.npy")
    esn_data = esn_data[0, 0]  # top mantissa, top exponent
    print(esn_data.shape)
    esn_plot = np.mean(esn_data, 0)[:500]
    esn_std = np.std(esn_data, 0)[:500]

    d2r2_data = np.load("d2r2_100/all_preds.npy")
    d2r2_data = d2r2_data[0, 0]  # top mantissa, top exponent
    d2r2_plot = np.mean(d2r2_data, 0)[:500]
    d2r2_std = np.std(d2r2_data, 0)[:500]

    inds = np.arange(500) / 200.0

    plt.plot(inds, lstm_plot_16, color="C0", label="LSTM 10 bit mantissa")
    plt.plot(inds, lstm_plot_32, color="C1", label="LSTM 23 bit mantissa")
    plt.plot(inds, esn_plot, color="C2", label="ESN %d bit mantissa" % bits)
    plt.plot(inds, d2r2_plot, color="C3", label="D2R2 %d bit mantissa" % bits)
    lstm_horizon_16 = np.argmax(lstm_plot_16 > 0.3)
    lstm_horizon_32 = np.argmax(lstm_plot_32 > 0.3)
    esn_horizon = np.argmax(esn_plot > 0.3)
    d2r2_horizon = np.argmax(d2r2_plot > 0.3)
    ticks = [0.0] + sorted([lstm_horizon_16 / 200.0, lstm_horizon_32 / 200.0, esn_horizon / 200.0, d2r2_horizon / 200.0]) + \
            [1.5, 2.0, 2.5]
    positions = list(ticks)
    # positions[1] -= 0.15
    print(lstm_horizon_16, lstm_horizon_32, esn_horizon, d2r2_horizon)
    print("%.2e, %.2e, %.2e, %.2e" % (lstm_horizon_16 / 200.0, lstm_horizon_32 / 200.0, esn_horizon / 200.0, d2r2_horizon / 200.0))
    print("%.2e, %.2e, %.2e, %.2e" % (lstm_std_16[lstm_horizon_16] / 200.0, lstm_std_32[lstm_horizon_32] / 200.0, esn_std[esn_horizon] / 200.0, d2r2_std[d2r2_horizon] / 200.0))
    print()
    plt.plot([lstm_horizon_16 / 200.0, lstm_horizon_16 / 200.0], [0, 0.3], color="C0", linewidth=0.75)
    plt.plot([lstm_horizon_32 / 200.0, lstm_horizon_32 / 200.0], [0, 0.3], color="C1", linewidth=0.75)
    plt.plot([esn_horizon / 200.0, esn_horizon / 200.0], [0, 0.3], color="C2", linewidth=0.75)
    plt.plot([d2r2_horizon / 200.0, d2r2_horizon / 200.0], [0, 0.3], color="C3", linewidth=0.75)
    plt.plot([0.3 for _ in range(500)], color="black", linewidth=0.75)
    plt.xticks(positions, ["%.1f" % x for x in ticks])
    # plt.legend(loc='upper right')
    plt.legend()
    plt.xlabel('Time step (MTU)')
    plt.ylabel('Relative (Norm) Error')
    # plt.title('Average error increase on Lorenz96')
    plt.xlim([0, 500 / 200.0])
    plt.ylim([0, 1.6])
    plt.savefig("avg_per_time_step.png")

    esn_data_exp_plot = np.load("ESN_exponents/plot_red.npy")[0, 5, :500]
    d2r2_data_exp_plot = np.load("D2R2_exponents/plot_red.npy")[0, 5, :500]

    plt.figure()
    plt.title("Exponent precision reduction")
    plt.plot(d2r2_data_exp_plot, color="C0", label="D2R2 4 bit exp")
    plt.plot(esn_data_exp_plot, color="C1", label="ESN 4 bit exp")
    plt.plot(esn_plot, color="C2", label="ESN 5 bit exp")
    plt.plot(d2r2_plot, color="C3", label="D2R2 5 bit exp")
    where_are_NaNs = np.isnan(d2r2_data_exp_plot)
    d2r2_data_exp_plot[where_are_NaNs] = 10.0
    d2r2_horizon_exp = np.argmax(d2r2_data_exp_plot > 0.3)
    esn_horizon_exp = np.argmax(esn_data_exp_plot > 0.3)
    print(d2r2_horizon_exp, esn_horizon_exp, esn_horizon, d2r2_horizon)
    print(d2r2_horizon_exp / 200.0, esn_horizon_exp / 200.0, esn_horizon / 200.0, d2r2_horizon / 200.0)
    plt.plot([d2r2_horizon_exp, d2r2_horizon_exp], [0, 0.3], color="C0", linewidth=0.75)
    plt.plot([esn_horizon_exp, esn_horizon_exp], [0, 0.3], color="C1", linewidth=0.75)
    plt.plot([esn_horizon, esn_horizon], [0, 0.3], color="C2", linewidth=0.75)
    plt.plot([d2r2_horizon, d2r2_horizon], [0, 0.3], color="C3", linewidth=0.75)
    plt.plot([0.3 for _ in range(500)], color="black", linewidth=0.75)
    plt.legend(loc='upper right')
    plt.xlabel('Time step (MTU)')
    plt.ylabel('Relative (Norm) Error')
    plt.xlim([0, 500])
    plt.ylim([0, 1.6])
    plt.savefig("avg_per_time_step_exp.png")


def plot_fourier(Eks, name):

    Ek = np.average(Eks, 0)
    # print(Ek)
    devs = np.std(Eks, 0)
    # print(devs)

    J = len(Ek)
    # print(Ek.shape)

    # Plotting the energy

    # plt.figure()
    # # plt.errorbar(range(0, J), Ek, devs, fmt="-o")
    # plt.stem(range(0, J), Ek, label="Energy")
    # for i, dev in enumerate(devs):
    #     plt.plot([i, i], [Ek[i] - dev, Ek[i] + dev], color="C1", linewidth=1.5)
    # plt.plot(range(0, J), Ek)
    # plt.xlabel("$Mode \; k$", fontsize=20)
    # plt.ylabel("$Energy \; E_k$", fontsize=20)
    # plt.legend(loc="upper right", fontsize=20)
    # plt.savefig(name + "_1.png")
    # plt.close()

    # only half the coefficients needed


    data = pd.read_csv('../../data_sets/lorenz_96.csv', header=None).values
    print(data.shape)
    comp_data = data[500000:500000 + 1000]
    lorenz_fourier = get_fourier(np.transpose(comp_data))
    print(lorenz_fourier.shape)
    lorenz_fourier = lorenz_fourier[:5]


    plt.figure()
    # plt.errorbar(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1], devs[0:int(J / 2) + 1], fmt="-o")
    plt.stem(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1], label="Model")
    plt.plot(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1])
    plt.plot(range(0, int(J / 2) + 1), lorenz_fourier, label="Lorenz")
    # for i, dev in enumerate(devs[0:int(J / 2) + 1]):
    #     plt.plot([i, i], [Ek[i] - dev, Ek[i] + dev], color="C1", linewidth=1.5)
    plt.xlabel("$Mode \; k$", fontsize=20)
    plt.ylabel("$Energy \; E_k$", fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.savefig(name + "_2.png")
    plt.close()

    # plotting number of modes used against cumulative energy %

    # idx = np.argsort(-Ek[0:int(J / 2) + 1])
    # # print(idx)
    # sEk = Ek[idx]
    # sDevs = devs[idx]
    # scale = np.sum(sEk)
    # cEk = np.cumsum(sEk / scale)
    # cDevs = np.cumsum(sDevs / scale)
    # plt.figure()
    # # plt.errorbar(range(0, int(J / 2) + 1), cEk, cDevs, fmt="-o")
    # plt.stem(range(0, int(J / 2) + 1), cEk)
    # plt.plot(range(0, int(J / 2) + 1), cEk)
    # for i, dev in enumerate(cDevs[0:int(J / 2) + 1]):
    #     plt.plot([i, i], [cEk[i] - dev, cEk[i] + dev], color="C1", linewidth=1.5)
    # plt.axhline(y=0.9, linewidth=0.5, color='r')
    # plt.xlabel("Number of most energetic modes used", fontsize=20)
    # plt.xticks(range(int(J / 2) + 1))
    # plt.ylabel("Cumulative energy %", fontsize=20)
    # # plt.legend(loc="upper right", fontsize=20)
    # plt.savefig(name + "_3.png")
    # plt.close()


def plot_fourier_3d(Eks, name):

    print(Eks.shape)
    Ek = np.average(Eks, 1)
    print(Ek.shape)
    # devs = np.std(Eks, 1)
    # print(devs.shape)

    J = Ek.shape[1]
    print(Ek.shape)
    bits = 10

    # Plotting the energy

    plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ticks = np.array([np.array(list(range(J))) for _ in range(bits)])
    z_ticks = np.array([np.ones((J, )) * x for x in range(bits)])
    ax.plot_wireframe(x_ticks, Ek, z_ticks)
    ax.set_xlabel("$Mode \; k$")
    ax.set_ylabel("$Energy \; E_k$")
    ax.set_zlabel("$Mantissa size \; bit_k$")
    ax.set_zticklabels(range(12, 2, -1))
    fig.savefig(name + "_1.png")
    plt.close()

    # only half the coefficients needed
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ticks = np.array([np.array(list(range(int(J / 2) + 1))) for _ in range(bits)])
    z_ticks = np.array([np.ones((int(J / 2) + 1, )) * x for x in range(bits)])
    ax.plot_wireframe(x_ticks, Ek[:, 0:int(J / 2) + 1], z_ticks)
    ax.set_xlabel("$Mode \; k$")
    ax.set_ylabel("$Energy \; E_k$")
    ax.set_zlabel("Mantissa size (bit)")
    ax.set_zticklabels(range(12, 2, -1))
    fig.savefig(name + "_2.png")
    plt.close()

    # plotting number of modes used against cumulative energy %

    idx = np.argsort(-Ek[:, 0:int(J / 2) + 1], axis=1)
    print(idx)
    sEk = np.zeros((bits, 5))
    for i in range(bits):
        sEk[i, :] = Ek[i, idx[i]]
    print(sEk.shape)
    scale = np.sum(sEk, axis=1)
    print(scale.shape)
    cEk = np.cumsum(sEk / scale[:, None], axis=1)
    print(cEk.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ticks = np.array([np.array(list(range(int(J / 2) + 1))) for _ in range(bits)])
    z_ticks = np.array([np.ones((int(J / 2) + 1,)) * x for x in range(bits)])
    ax.plot_wireframe(x_ticks, cEk, z_ticks)
    ax.axhline(y=0.9, linewidth=0.5, color='r')
    plt.xticks(range(int(J / 2) + 1))
    ax.set_xlabel("Number of most energetic modes used")
    ax.set_ylabel("Cumulative energy %")
    ax.set_zlabel("Mantissa size (bit)")
    ax.set_zticklabels(range(12, 2, -1))
    ax.set_ylim(0, 1)
    # plt.legend(loc="upper right", fontsize=20)
    fig.savefig(name + "_3.png")
    plt.close()


def plot_fourier_grid(Eks1, Eks2, name):

    print("grid")
    print(Eks1.shape)
    Ek1 = np.average(Eks1, 1)
    devs1 = np.std(Eks1, 1)
    print(Ek1.shape)
    print(devs1.shape)
    Ek2 = np.average(Eks2, 1)
    devs2 = np.std(Eks2, 1)
    print(Ek2.shape)
    print(devs2.shape)
    Ek = np.array([Ek1, Ek2])
    devs = np.array([devs1, devs2])
    print(Ek.shape)
    print(devs.shape)

    J = int(Ek1.shape[1] / 2) + 1
    print(J)
    bits = Ek1.shape[0]
    print(bits)

    cols = ["D2R2", "ESN"]
    rows = ['Mantissa: {} bits'.format(col) for col in range(12, 2, -1)]

    plt.figure()
    fig, axs = plt.subplots(ncols=2, nrows=bits, figsize=(2 * 7, bits * 4))
    fig.suptitle("D2R2 and ESN Fourier modes")

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for i in range(bits):
        for j in range(2):
            ax = axs[i, j]
            ax.stem(range(0, J), Ek[j, i, 0:J], label="Energy")
            ax.plot(range(0, J), Ek[j, i, 0:J])
            for k, dev in enumerate(devs[j, i, 0:J]):
                ax.plot([k, k], [Ek[j, i, k] - dev, Ek[j, i, k] + dev], color="C1", linewidth=1.5)
            if i == bits - 1:
                ax.set_xlabel("$Mode \; k$")
            if j == 0:
                ax.set_ylabel("$Energy \; E_k$", rotation=90)
            ax.legend(loc="upper right")

    plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.02)
    plt.savefig(name + ".png")
    plt.close()


def plot_fourier_pair(Eks1, Eks2, name):

    print("grid")
    print(Eks1.shape)
    Ek1 = np.average(Eks1, 1)
    devs1 = np.std(Eks1, 1)
    print(Ek1.shape)
    print(devs1.shape)
    Ek2 = np.average(Eks2, 1)
    devs2 = np.std(Eks2, 1)
    print(Ek2.shape)
    print(devs2.shape)
    Ek = np.array([Ek1, Ek2])
    devs = np.array([devs1, devs2])
    print(Ek.shape)
    print(devs.shape)

    J = int(Ek1.shape[1] / 2) + 1
    print(J)
    bits = Ek1.shape[0]
    print(bits)

    plt.figure()
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(2 * 7, 7))
    fig.suptitle("D2R2 and ESN Fourier modes")

    axs[0].set_title("D2R2")
    axs[1].set_title("ESN")


    for i in range(bits):
        for j in range(2):
            ax = axs[j]
            ax.plot(range(0, J), Ek[j, i, 0:J], color="C%d" % i, label="mantissa bits: %d" % (i + 3))
            if i == bits - 1:
                ax.set_xlabel("$Mode \; k$")
            if j == 0:
                ax.set_ylabel("$Energy \; E_k$", rotation=90)
            ax.legend(loc="upper right")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    plt.savefig(name + ".png")
    plt.close()


def main6():
    mantissas = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    exponents = [10]

    # d2r2
    d2r2_data64 = np.load("d2r2_100/fourier64.npy")
    d2r2_data32 = np.load("d2r2_100/fourier32.npy")
    d2r2_data16 = np.load("d2r2_100/fourier16.npy")
    d2r2_data = np.load("d2r2_100/fourier.npy")
    plot_fourier(d2r2_data64, "d2r2_100/fourier_64")
    plot_fourier(d2r2_data32, "d2r2_100/fourier_32")
    plot_fourier(d2r2_data16, "d2r2_100/fourier_16")
    for i, mantissa in enumerate(mantissas):
        for j, exponent in enumerate(exponents):
            plot_fourier(d2r2_data[i, j], "d2r2_100/fourier_%dm%de" % (mantissa, exponent))

    # plot_fourier_3d(d2r2_data[:, 0, :], "d2r2_stack")

    # esn
    esn_data64 = np.load("esn_100/fourier64.npy")
    esn_data32 = np.load("esn_100/fourier32.npy")
    esn_data16 = np.load("esn_100/fourier16.npy")
    esn_data = np.load("esn_100/fourier.npy")
    plot_fourier(esn_data64, "esn_100/fourier_64")
    plot_fourier(esn_data32, "esn_100/fourier_32")
    plot_fourier(esn_data16, "esn_100/fourier_16")
    for i, mantissa in enumerate(mantissas):
        for j, exponent in enumerate(exponents):
            plot_fourier(esn_data[i, j], "esn_100/fourier_%dm%de" % (mantissa, exponent))

    # plot_fourier_3d(esn_data[:, 0, :], "esn_stack")

    # plot_fourier_grid(d2r2_data[:, 0, :], esn_data[:, 0, :], "fourier_grid")
    # plot_fourier_pair(d2r2_data[:, 0, :], esn_data[:, 0, :], "fourier_pair")


if __name__ == "__main__":
    # main1()
    # main2()
    # main3()
    main4()
    # main5()  # exp vs d2r2 vs esn | lstm esn d2r2
    # main6()
