import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


from ESN_Paper_Code_Table_min import get_fourier


def draw_plot(data, filename, name):
    p_limit = 0.05
    errors = data[:, 0]
    p = data[:, 1]
    limit_id = np.argmax(p >= p_limit)
    err1 = errors[0:limit_id]
    err2 = errors[limit_id - 1:]
    plt.figure()
    plt.title(name + " vs Lorenz96 Fourier modes Pearson's correlation")
    plt.ylabel("Correlation")
    plt.xlabel("Mantissa size (bit)")
    plt.axhline(0, color='black')
    plt.plot(range(len(err1)), err1, linestyle="-", color="C0", label="p-value <= 0.05")
    plt.plot(range(len(err1) - 1, len(errors)), err2, linestyle="--", color="C0", label="p value > 0.05")
    plt.xticks(range(10), range(12, 2, -1))
    plt.ylim((-1.2, 1.2))
    plt.grid()
    plt.legend()
    plt.savefig(filename + ".png")


def main():
    training_len = 500000
    offsets = np.arange(1000, 1000 + 100 * 1000, 1000)
    print(offsets)
    print(offsets + training_len)
    assert offsets.shape[0] == 100

    data = pd.read_csv('../../data_sets/lorenz_96.csv', header=None).values
    print(data.shape)

    d2r2_data = np.load("d2r2_100/fourier.npy")
    esn_data = np.load("esn_100/fourier.npy")

    mantissas = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    exponents = [10]

    d2r2_cor = np.zeros((len(mantissas), len(exponents), len(offsets), 2))
    esn_cor = np.zeros((len(mantissas), len(exponents), len(offsets), 2))

    for i, mantissa in enumerate(mantissas):
        for j, exponent in enumerate(exponents):
            for k, offset in enumerate(offsets + training_len):
                comp_data = data[offset:offset + 1000]
                lorenz_fourier = get_fourier(np.transpose(comp_data))
                d2r2_cor[i, j, k, :] = pearsonr(lorenz_fourier[:5].flatten(), d2r2_data[i, j, k, :5].flatten())
                esn_cor[i, j, k, :] = pearsonr(lorenz_fourier[:5].flatten(), esn_data[i, j, k, :5].flatten())

    avg_d2r2_cor = np.average(d2r2_cor, 2)
    print(avg_d2r2_cor.shape)
    print(avg_d2r2_cor)
    draw_plot(avg_d2r2_cor[:, 0], "d2r2_parson", "D2R2")

    avg_esn_cor = np.average(esn_cor, 2)
    print(avg_esn_cor.shape)
    print(avg_esn_cor)
    draw_plot(avg_esn_cor[:, 0], "esn_parson", "ESN")


    p_limit = 0.05
    plt.figure()
    # plt.title(" vs Lorenz96 Fourier modes Pearson's correlation")
    plt.ylabel("Correlation")
    plt.xlabel("Mantissa size (bit)")
    plt.axhline(0, color='black')

    errors = avg_d2r2_cor[:, 0, 0]
    p = avg_d2r2_cor[:, 0, 1]
    limit_id = np.argmax(p >= p_limit)
    err1 = errors[0:limit_id]
    err2 = errors[limit_id - 1:]
    plt.plot(range(len(err1)), err1, linestyle="-", color="C0", label="D2R2, p-value <= 0.05")
    plt.plot(range(len(err1) - 1, len(errors)), err2, linestyle="--", color="C0", label="D2R2, p value > 0.05")

    errors = avg_esn_cor[:, 0, 0]
    p = avg_esn_cor[:, 0, 1]
    limit_id = np.argmax(p >= p_limit)
    err1 = errors[0:limit_id]
    err2 = errors[limit_id - 1:]
    plt.plot(range(len(err1)), err1, linestyle="-", color="C1", label="ESN, p-value <= 0.05")
    plt.plot(range(len(err1) - 1, len(errors)), err2, linestyle="--", color="C1", label="ESN, p value > 0.05")

    plt.xticks(range(10), range(12, 2, -1))
    plt.ylim((-1.2, 1.2))
    plt.grid()
    plt.legend()
    plt.savefig("combined_parson.png")


if __name__ == "__main__":
    main()
