import numpy as np
import numpy.matlib as matlib
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import matplotlib.pyplot as plt
from math import frexp

from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

# global variables
# This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

approx_res_size = 4000

model_params = {
    'tau': 0.25,
    'nstep': 1000,
    'N': 8,
    'd': 22
}

res_params = {
    'radius': .1,  # original: .1, pathak: 1.2
    'degree': 3,  # original 3, pathak 6
    'sigma': 0.5,  # original .5, pathak .1
    'train_length': 300000,
    'N': int(np.floor(approx_res_size / model_params['N']) * model_params['N']),
    'num_inputs': model_params['N'],
    'predict_length': 1000,
    'beta': 0.0001
}


def get_fourier(vals):

    vals = np.transpose(vals)
    J = vals.shape[1]  # the number of variables
    # print(J)

    dt = 0.005  # integration time step
    # N_train = 500000  # number of points for training
    N_train = vals.shape[0]

    T = int(N_train * dt)  # time interval (seconds) of training

    # create X (for training) and X_test

    X = vals

    # compute mean of X (Xbar in Equation 4.2)
    X_bar = X.mean()

    # to implement the integral from T_0 to T_0 + T in equation 4.2 (part 3)
    time_vector = np.transpose(np.matrix(range(1, N_train + 1))) * dt

    # Calculate Ep from Equation 4.2 (see the structure of the computation following the equation)
    Ep = np.sum(np.trapz(np.power(X[0:N_train, :] - X_bar, 2), time_vector, axis=0)) / (2. * T)
    #print("Variance in energy fluctuations {:f}".format(Ep))

    # normalization step (scaling Xj and dt according to the first two parts of Equation 4.2)

    Xj = (X - X_bar) / np.sqrt(Ep)
    # Xj_bar = np.mean(Xj)

    # Ep_norm = np.sum(np.trapz(np.power(Xj - Xj_bar, 2), time_vector, axis=0)) / (2. * T)
    # print("Normalized average variance in energy fluctuations (MUST BE ONE): {:f}".format(Ep_norm))
    # print(Ep_norm)

    # Discrete Fourier Coefficients of the normalized X: these are Xk in Equation 4.4
    Xk = np.fft.fft(Xj, J, 1) / J

    # FFT test  (check that the fft works by reconstructing Xj from Xk)
    # Xj_rec = np.real(J * np.fft.ifft(Xk, J, 1))
    # print("Fourier transformation reconstruction error (MUST BE ZERO):  {:f}".format(np.linalg.norm(Xj - Xj_rec)))
    #print(Xk.shape)

    # Average Xk
    Xk_avg = np.mean(Xk, 0)

    # Energy spectrum  (using the equation below Equation 4.4 defining Ek)
    Xkc = Xk - matlib.repmat(Xk_avg, N_train, 1)
    Ek = np.real(np.mean(np.multiply(Xkc, np.conj(Xkc)), 0))
    #print(Ek.shape)

    return Ek


def plot_fourier(Eks, name):

    Ek = np.average(Eks, 0)
    print(Ek)
    devs = np.std(Eks, 0)
    print(devs)

    J = len(Ek)
    print(Ek.shape)

    # Plotting the energy

    plt.figure()
    # plt.errorbar(range(0, J), Ek, devs, fmt="-o")
    plt.stem(range(0, J), Ek, label="Energy")
    for i, dev in enumerate(devs):
        plt.plot([i, i], [Ek[i] - dev, Ek[i] + dev], color="C1", linewidth=1.5)
    plt.plot(range(0, J), Ek)
    plt.xlabel("$Mode \; k$", fontsize=20)
    plt.ylabel("$Energy \; E_k$", fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.savefig(name + "_1.png")
    plt.close()

    # only half the coefficients needed
    plt.figure()
    # plt.errorbar(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1], devs[0:int(J / 2) + 1], fmt="-o")
    plt.stem(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1], label="Energy")
    plt.plot(range(0, int(J / 2) + 1), Ek[0:int(J / 2) + 1])
    for i, dev in enumerate(devs[0:int(J / 2) + 1]):
        plt.plot([i, i], [Ek[i] - dev, Ek[i] + dev], color="C1", linewidth=1.5)
    plt.xlabel("$Mode \; k$", fontsize=20)
    plt.ylabel("$Energy \; E_k$", fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.savefig(name + "_2.png")
    plt.close()

    # plotting number of modes used against cumulative energy %

    idx = np.argsort(-Ek[0:int(J / 2) + 1])
    print(idx)
    sEk = Ek[idx]
    sDevs = devs[idx]
    scale = np.sum(sEk)
    cEk = np.cumsum(sEk / scale)
    cDevs = np.cumsum(sDevs / scale)
    plt.figure()
    # plt.errorbar(range(0, int(J / 2) + 1), cEk, cDevs, fmt="-o")
    plt.stem(range(0, int(J / 2) + 1), cEk)
    plt.plot(range(0, int(J / 2) + 1), cEk)
    for i, dev in enumerate(cDevs[0:int(J / 2) + 1]):
        plt.plot([i, i], [cEk[i] - dev, cEk[i] + dev], color="C1", linewidth=1.5)
    plt.axhline(y=0.9, linewidth=0.5, color='r')
    plt.xlabel("Number of most energetic modes used", fontsize=20)
    plt.xticks(range(int(J / 2) + 1))
    plt.ylabel("Cumulative energy %", fontsize=20)
    # plt.legend(loc="upper right", fontsize=20)
    plt.savefig(name + "_3.png")
    plt.close()


def generate_reservoir(size, radius, degree):
    sparsity = degree / float(size)
    a = sparse.rand(size, size, density=sparsity).todense()
    values = np.linalg.eigvals(a)
    e = np.max(np.abs(values))
    a = (a / e) * radius
    return a


def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'], res_params['train_length']))
    for i in range(res_params['train_length'] - 1):
        if i % (res_params['train_length'] / 100) == 0:
            print("\rReservoir layer %d / %d" % (i, res_params['train_length'] - 1), end="")
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input[:, i]))
    print()
    return states


def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N'] / res_params['num_inputs'])
    Win = np.zeros((res_params['N'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        print("\rTrain reservoir %d / %d" % (i, res_params['num_inputs']), end="")
        np.random.seed(seed=i)
        Win[i * q: (i + 1) * q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])
    print()

    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win, states


def train(res_params, states, data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2, np.shape(states2)[0] - 2):
        print("\rTrain %d / %d" % (j, np.shape(states2)[0] - 2), end="")
        if np.mod(j, 2) == 0:
            states2[j, :] = (states[j - 1, :] * states[j - 2, :]).copy()
    print()
    U = np.dot(states2, states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(states2, data.transpose()))
    return Wout.transpose()


def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'], res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = (x[j - 1] * x[j - 2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout, x_aug)))
        output[:, i] = out
        x1 = np.tanh(np.dot(A, x) + np.dot(Win, out))
        x = np.squeeze(np.asarray(x1))
    return output, x


def predictTF(A, Win, x, Wout, states):
    output = np.zeros((states.shape[0], states.shape[1]))
    for i in range(states.shape[1]):
        x_aug = x.copy()
        for j in range(2, np.shape(x_aug)[0] - 2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = (x[j - 1] * x[j - 2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout, x_aug)))
        output[:, i] = out
        x1 = np.tanh(np.dot(A, x) + np.dot(Win, states[:,i]))
        x = np.squeeze(np.asarray(x1))
    return output, x


##### New poly functions below

def polypred(state, order):  # Given a state vector, output its polynomial expansion 1,x,x^2, order must be 2,3,or 4
    N = len(state)
    size = 1 + N  # 0 order will have 1 term, 1 order will have N terms
    for i in range(2, order + 1):
        comb = combinations_with_replacement(np.arange(N), i)
        combos = list(comb)
        size += len(combos)
    polyexp = np.zeros(size)
    polyexp[0] = 1
    polyexp[1:N + 1] = state[:]
    comb = combinations_with_replacement(np.arange(N), 2)
    combos = list(comb)
    for i in range(len(combos)):
        polyexp[N+1+i] = state[combos[i][0]] * state[combos[i][1]]
    if order > 2:
        comb3 = combinations_with_replacement(np.arange(N), 3)
        combos3 = list(comb3)
        for j in range(len(combos3)):
            polyexp[N + 2 + i + j] = state[combos3[j][0]] * state[combos3[j][1]] * state[combos3[j][2]]
    if order > 3:
        comb4 = combinations_with_replacement(np.arange(N), 4)
        combos4 = list(comb4)
        for k in range(len(combos4)):
            polyexp[N + 3 + i + j + k] = state[combos4[k][0]] * state[combos4[k][1]] * state[combos4[k][2]] * state[combos4[k][3]]
    return polyexp


def polyfeat(states, order): # Given state vector, turn it into feature vector to fit linear regression with
    # We are using their convention that features[0] = 0, features[1] = polypred(states[0]) etc
    N = np.shape(states)[0]
    size = 1 + N  # 0 order will have 1 term, 1 order will have N terms
    for i in range(2, order + 1):
        comb = combinations_with_replacement(np.arange(N), i)
        combos = list(comb)
        size += len(combos)
    polyfeatures = np.zeros((size, np.shape(states)[1]))
    pp = PolynomialFeatures(degree=order)
    for i in range(np.shape(states)[1]):
        polyfeatures[:, i] = pp.fit_transform(states[:, i].reshape(1, -1))
    return polyfeatures


def polytrain(features, beta, size, data):
    idenmat = beta * sparse.identity(size)
    U = np.dot(features, features.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(features, data.transpose()))
    return Wout.transpose()


def polyauto(startstate, Wout, order, autotime):
    N = len(startstate)
    state = startstate
    predictions = np.zeros((N, autotime))
    for i in range(autotime):
        polyfeatures = polypred(state, order)
        state = np.array(np.dot(Wout, polyfeatures)).reshape(N)
        predictions[:, i] = state
    return predictions


def reduce_precision(value, mantissa_size, exponent_size):

    mantissa, exponent = frexp(value)
    sign = 1 if mantissa >= 0 else -1

    reduced_exponent = exponent
    if exponent < -(2 ** (exponent_size - 1)):
        return sign * 0.5 * (2.0 ** (-(2 ** (exponent_size - 1))))
    elif exponent > (2 ** (exponent_size - 1)) - 1:
        return sign * 0.9 * (2.0 ** ((2 ** (exponent_size - 1)) - 1))
    elif mantissa < -0.00001 or mantissa > 0.00001:
        mantissa = abs(mantissa)
        rem_mantissa = mantissa - 0.5
        fraction = 1 / 4
        for i in range(mantissa_size):
            if rem_mantissa > fraction:
                rem_mantissa -= fraction
            fraction /= 2
        reduced_mantissa = mantissa - rem_mantissa
        reduced_value = sign * reduced_mantissa * (2.0 ** reduced_exponent)
        return reduced_value
    else:
        return value


def main():
    reduce_precision_vec = np.vectorize(reduce_precision)

    dataf = pd.read_csv('../../data_sets/lorenz_96.csv', header=None)  # dataf = pd.read_csv('.\Pytorch\ESN\data_sets\\threetier_lorenz_v3.csv', header=None)
    data = np.transpose(np.array(dataf))  # data is 8x1M

    # For each of 10 different starting positions, do a TF and AUTO run, and store final (mean) normed errors
    # offsets = np.array([0, 5000, 25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000])

    offsets = np.arange(1000, 1000 + 100 * 1000, 1000)
    assert offsets.shape[0] == 100

    # offsets = np.arange(1000, 1000 + 20 * 1000, 1000)
    # assert offsets.shape[0] == 20

    do_fourier = True

    mantissas = [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    exponents = [10]

    # mantissas = [32, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    # exponents = [10]

    # mantissas = [12]
    # exponents = [9, 8, 7, 6, 5, 4, 3]

    d2r2 = False
    esn = True

    tabledat = np.zeros((len(offsets), ))
    tabledat32 = np.zeros((len(offsets), ))
    tabledat16 = np.zeros((len(offsets), ))

    fourier_data64 = np.zeros((len(offsets), 8))
    fourier_data32 = np.zeros((len(offsets), 8))
    fourier_data16 = np.zeros((len(offsets), 8))
    fourier_data = np.zeros((len(mantissas), len(exponents), len(offsets), 8))

    pred_plot_data = np.zeros((len(mantissas), len(exponents), len(offsets), 1000))

    W32prec = np.single
    W16prec = np.half
    # Start Data
    sd32prec = np.double
    sd16prec = np.double
    # Win
    Win32prec = np.single
    Win16prec = np.half
    # A
    A32prec = np.single
    A16prec = np.half
    # Start State
    ss32prec = np.single
    ss16prec = np.half

    if esn:
        x, Wout, A, Win, trainstates = train_reservoir(res_params, data[:, shift_k:shift_k + res_params['train_length']])
        print("Training Done")

        Wout32 = np.copy(Wout).astype(W32prec)
        Wout16 = np.copy(Wout).astype(W16prec)
        Win32 = np.copy(Win).astype(Win32prec)
        Win16 = np.copy(Win).astype(Win16prec)
        A32 = np.copy(A).astype(A32prec)
        A16 = np.copy(A).astype(A16prec)


        table_dat = np.zeros((len(mantissas), len(exponents), len(offsets)))
        plots = np.zeros((len(mantissas), len(exponents), 1000))
        for i in range(len(offsets)):

            print("Offsets ", i, "/", len(offsets))

            start_time = shift_k + res_params['train_length'] + offsets[i]

            for j, mantissa in enumerate(mantissas):
                for k, exponent in enumerate(exponents):
                    w_out_red = reduce_precision_vec(Wout, mantissa, exponent)
                    w_in_red = reduce_precision_vec(Win, mantissa, exponent)
                    a_red = reduce_precision_vec(A, mantissa, exponent)

                    _, start_state = predictTF(a_red, w_in_red, x * 0, w_out_red, data[:, start_time - 50:start_time])
                    start_state_red = reduce_precision_vec(start_state, mantissa, exponent)

                    output_red, _ = predict(a_red, w_in_red, res_params, start_state_red, w_out_red)
                    auto_err_red = output_red - data[:, start_time + 3:start_time + 1003]
                    auto_err_e_red = np.linalg.norm(auto_err_red, axis=0) / np.linalg.norm(data[:, start_time + 2:start_time + 1002], axis=0)
                    pred_plot_data[j, k, i, :] = auto_err_e_red
                    table_dat[j, k, i] = np.where(auto_err_e_red > .3)[0][0]

                    if i == 0:  # save first set of auto predictions in each to plot
                        plots[j, k] = np.copy(auto_err_e_red)


    print(table_dat)
    np.save("1_red_bit.npy", table_dat)
    np.save("1_plot_red.npy", np.array(plots))
    np.save("1_mantissas.npy", np.array(mantissas))
    np.save("1_exponents.npy", np.array(exponents))
    np.save("1_all_preds.npy", pred_plot_data)
    
    





if __name__ == "__main__":
    main()
