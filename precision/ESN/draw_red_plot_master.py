import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ESN_Paper_Code_Table_min import get_fourier


def main2_plt(data, mantissas, filename, name, x_name):
    plt.figure()
    plt.xlabel(x_name + ' size (bits)')
    plt.ylabel('Horizon with standard deviation (MTU)')
    plt.title(name + ' varying precision on Lorenz96')
    plt.errorbar(range(len(mantissas), 0, -1), data[:, 0], data[:, 1], fmt='-o')
    plt.xticks(range(len(mantissas), 0, -1), mantissas)
    plt.grid(linestyle='--')
    plt.savefig(filename + ".png")


def main2():


    d2r2_data_all = np.array([
        # np.array([251.43, 89.73118243]),
        # np.array([251.45, 89.78378194]),
        # np.array([251.45, 89.79759184]),
        # np.array([251.6 , 90.35286382]),
        # np.array([254.39, 105.98212418]),
        np.array([252.0 , 93.69738524]),
        np.array([249.59, 87.48555252]),
        np.array([252.23, 93.2638038 ]),
        np.array([253.21, 96.1125689 ]),
        np.array([260.17, 105.22908866]),
        np.array([254.45, 98.71184073]),
        np.array([227.19, 86.37264555]),
        np.array([209.42, 74.94306906]),
        np.array([162.91, 67.71042682]),
        np.array([94.28, 39.13389324]),
        np.array([64.34, 27.11465287]),
        np.array([49.7 , 18.83427726]),
        np.array([28.82, 9.35882471]),
        np.array([14.16, 4.79107504]),
        np.array([10.09, 3.5639725 ]),
        np.array([4.3 , 1.15325626]),
        np.array([2.21, 0.6677574])
    ]) / 200
    mantissas = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    main2_plt(d2r2_data_all, mantissas, "esn_all_mean_std", "ESN", "Mantissa")


if __name__ == "__main__":
    main2()
