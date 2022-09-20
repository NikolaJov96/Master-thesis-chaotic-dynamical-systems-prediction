# Version of ESN_Paper_Code meant to generate the table Ankit requested
# For each of the 3 methods (esn/pathak esn/poly regression):
# Mean +/- sd over 10 runs:
# Mean training error, mean testing error (both TF and AUTO), and time till norm error hits .3 (in MTU, which is 200 timesteps)
# Except training error, which will only have one run

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd

from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

# global variables
# This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

# out of memory error with 5k. Try 2.5k
approx_res_size = 5000//2
approx_res_size = 1500

model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 8,
                'd': 22}

res_params = {'radius': .1, #original: .1, pathak: 1.2
              'degree': 3, # original 3, pathak 6
              'sigma': 0.5, # original .5, pathak .1
              'train_length': 300000,
              'N': int(np.floor(approx_res_size / model_params['N']) * model_params['N']),
              'num_inputs': model_params['N'],
              'predict_length': 2000,
              'beta': 0.0001
              }


# The ESN functions for training
def generate_reservoir(size, radius, degree):
    sparsity = degree / float(size);
    A = sparse.rand(size, size, density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A / e) * radius
    return A


def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'], res_params['train_length']))
    for i in range(res_params['train_length'] - 1):
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input[:, i]))
    return states


def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N'] / res_params['num_inputs'])
    Win = np.zeros((res_params['N'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i * q: (i + 1) * q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])

    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win, states


def train(res_params, states, data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2, np.shape(states2)[0] - 2):
        if (np.mod(j, 2) == 0):
            states2[j, :] = (states[j - 1, :] * states[j - 2, :]).copy()
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

# original predict function only works for AUTO, where we provide initial state x
# So we also need a TF predictor, and something to get initial state
# Actually, predictTF can be done for both - first provide previous~50 states (with xin = 0) and use x output...
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

def polypred(state, order): # Given a state vector, output its polynomial expansion 1,x,x^2, order must be 2,3,or 4
    N = len(state)
    size = 1+N # 0 order will have 1 term, 1 order will have N terms
    for i in range(2,order+1):
        comb = combinations_with_replacement(np.arange(N), i)
        combos = list(comb)
        size+=len(combos)
    polyexp = np.zeros(size)
    polyexp[0] = 1
    polyexp[1:N+1] = state[:]
    comb = combinations_with_replacement(np.arange(N), 2)
    combos = list(comb)
    for i in range(len(combos)):
        polyexp[N+1+i] = state[combos[i][0]] * state[combos[i][1]]
    if order > 2:
        comb3 = combinations_with_replacement(np.arange(N), 3)
        combos3 = list(comb3)
        for j in range(len(combos3)):
            polyexp[N+2+i+j] = state[combos3[j][0]] * state[combos3[j][1]] * state[combos3[j][2]]
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
    polyfeatures = np.zeros((size,np.shape(states)[1]))
    #for i in range(np.shape(states)[1]-1):
    #    polyfeatures[:,i+1] = polypred(states[:,i],order)
    #return polyfeatures

    #polyfeatures[:,i] = pp.fit_transform(states[:,i].reshape(1, -1)) # no longer off by one - target must now be off by one
    #return polyfeatures
    pp = PolynomialFeatures(degree=order)
    #polyfeatures2 = (pp.fit_transform(states.transpose())).transpose()
    #return polyfeatures2
    for i in range(np.shape(states)[1]):
        polyfeatures[:, i] = pp.fit_transform(states[:, i].reshape(1,-1))
    return polyfeatures

def polytrain(features, beta, size, data):
    idenmat = beta * sparse.identity(size)
    U = np.dot(features, features.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(features, data.transpose()))
    return Wout.transpose()

def polyauto(startstate,Wout,order, autotime):
    N = len(startstate)
    state = startstate
    predictions = np.zeros((N, autotime))
    for i in range(autotime):
        polyfeatures = polypred(state,order)
        state = np.array(np.dot(Wout,polyfeatures)).reshape(N)
        predictions[:,i] = state
    return predictions

def polytf(states,Wout, order):
    N = states.shape[0]
    tftime = states.shape[1]
    predictions = np.zeros((N,tftime))
    for i in range(tftime):
        polyfeatures = polypred(states[:,i], order)
        state = np.array(np.dot(Wout, polyfeatures)).reshape(N)
        predictions[:, i] = state
    return predictions


dataf = pd.read_csv('../../data_sets/lorenz_96.csv', header=None)  # dataf = pd.read_csv('.\Pytorch\ESN\data_sets\\threetier_lorenz_v3.csv', header=None)
data = np.transpose(np.array(dataf)) # data is 8x1M

import matplotlib.pyplot as plt

W32prec = np.single
W16prec = np.half
# Poly Features
pf32prec = np.double
pf16prec = np.double
# Start Data
sd32prec = np.double
sd16prec = np.double

polyrun = True
if polyrun:
    offset = 1# set to 1 if using Devika new formulation
    order = 4 # 2 to 4 for quadratic to quartic
    size = 1+8
    for i in range(2, order + 1):
        comb = combinations_with_replacement(np.arange(8), i)
        combos = list(comb)
        size += len(combos)
    polyfeatures = polyfeat(data[:, shift_k:shift_k + res_params['train_length']],order)
    print('Poly Feature Done')
    Wout = polytrain(polyfeatures,res_params['beta'],size,data[:, shift_k+offset:shift_k+offset + res_params['train_length']])
    print('Poly Wout Done')
    # exit(0) # stop
    # Get train error - its just features * Wout

    # New - Krishna wants dropping terms or orders
    #Wout32 = np.copy(Wout)
    #Wout32[np.abs(Wout32)<np.percentile(np.abs(Wout32), 15)] = 0 # bottom 15% set to 0
    #Wout16 = np.copy(Wout)
    #Wout16[np.abs(Wout16)<np.percentile(np.abs(Wout16), 20)] = 0 # bottom 20% set to 0

    # Apply lower (bit) precision if interested
    Wout32 = np.copy(Wout).astype(W32prec)
    Wout16 = np.copy(Wout).astype(W16prec)
    polyfeatures32 = np.copy(polyfeatures).astype(pf32prec)
    polyfeatures16 = np.copy(polyfeatures).astype(pf16prec)

    print("Step 1")

    trainout = Wout @ polyfeatures
    trainerr = trainout - data[:,shift_k+offset:shift_k+offset+res_params['train_length']]
    trainerre = np.linalg.norm(trainerr,axis=0) / np.linalg.norm(data[:,shift_k+offset:shift_k+offset+res_params['train_length']], axis = 0)

    print("Step 2")

    trainout32 = Wout32 @ polyfeatures32
    trainerr32 = trainout32 - data[:, shift_k + offset:shift_k + offset + res_params['train_length']]
    trainerre32 = np.linalg.norm(trainerr32, axis=0) / np.linalg.norm(
        data[:, shift_k + offset:shift_k + offset + res_params['train_length']], axis=0)

    print("Step 3")

    trainout16 = Wout16 @ polyfeatures16
    trainerr16 = trainout16 - data[:, shift_k + offset:shift_k + offset + res_params['train_length']]
    trainerre16 = np.linalg.norm(trainerr16, axis=0) / np.linalg.norm(
        data[:, shift_k + offset:shift_k + offset + res_params['train_length']], axis=0)

    print("Step 4")

    # For each of 10 different starting positions, do a TF and AUTO run, and store final (mean) normed errors
    # offsets = np.array([0,5000,25000,50000,75000,100000,150000,200000,250000,300000])
    offsets = np.array([0])
    tabledat = np.zeros((3,len(offsets)))
    tabledat32 = np.zeros((3, len(offsets)))
    tabledat16 = np.zeros((3, len(offsets)))
    for i in range(len(offsets)):

        print("Offsets ", i, "/", len(offsets))

        starttime = shift_k + res_params['train_length']+offsets[i]
        startdat = np.copy(data[:,starttime:starttime+2000])
        startdat32 = np.copy(startdat).astype(sd32prec)
        startdat16 = np.copy(startdat).astype(sd16prec)
        polytfpred = polytf(startdat,Wout,order)
        polyautopred = polyauto(startdat[:,0],Wout,order,2000)
        polytferr = polytfpred - data[:,starttime+1:starttime+2001]
        polyautoerr = polyautopred - data[:,starttime+1:starttime+2001]
        polytferre = np.linalg.norm(polytferr,axis = 0) / np.linalg.norm(data[:,starttime+1:starttime+2001], axis = 0)
        polyautoerre = np.linalg.norm(polyautoerr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat[0,i] = np.mean(polytferre)
        tabledat[1,i] = np.mean(polyautoerre)
        tabledat[2,i] = np.where(polyautoerre>.3)[0][0] # raw number of timesteps - not divided by 200 yet

        print("Offsets 1")

        # 32 version
        polytfpred32 = polytf(startdat32, Wout32, order)
        polyautopred32 = polyauto(startdat32[:, 0], Wout32, order, 2000)
        polytferr32 = polytfpred32 - data[:, starttime + 1:starttime + 2001]
        polyautoerr32 = polyautopred32 - data[:, starttime + 1:starttime + 2001]
        polytferre32 = np.linalg.norm(polytferr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        polyautoerre32 = np.linalg.norm(polyautoerr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat32[0, i] = np.mean(polytferre32)
        tabledat32[1, i] = np.mean(polyautoerre32)
        tabledat32[2, i] = np.where(polyautoerre32 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet

        print("Offsets 2")

        # 16 version
        polytfpred16 = polytf(startdat16, Wout16, order)
        polyautopred16 = polyauto(startdat16[:, 0], Wout16, order, 2000)
        polytferr16 = polytfpred16 - data[:, starttime + 1:starttime + 2001]
        polyautoerr16 = polyautopred16 - data[:, starttime + 1:starttime + 2001]
        polytferre16 = np.linalg.norm(polytferr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        polyautoerre16 = np.linalg.norm(polyautoerr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat16[0, i] = np.mean(polytferre16)
        tabledat16[1, i] = np.mean(polyautoerre16)
        tabledat16[2, i] = np.where(polyautoerre16 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet
        if(i == 0): # save first set of auto predictions in each to plot
            pplot = np.copy(polyautoerre)
            pplot32 = np.copy(polyautoerre32)
            pplot16 = np.copy(polyautoerre16)
            sumplot = np.copy(polyautoerre)
        else:
            sumplot += polyautoerre
    sumplot /= i

    fig = plt.figure()
    print(pplot[200:210])
    print(pplot32[200:210])
    print(pplot16[200:210])
    plt.plot(pplot)
    plt.plot(pplot32)
    plt.plot(pplot16)
    plt.legend(['64 bit', '32 bit', '16 bit'])
    plt.xlabel('Testing Step')
    plt.ylabel('Relative (Norm) Error')
    plt.title('Lorenz96 for varying precision')
    plt.xlim([0,500])
    plt.ylim([0,1.6])
    plt.savefig("pre1.png")  # plt.show()
    exit(0)

    # plot of histogram of abs(Wout)

    bins = 10
    print("bins: ", bins)

    # plt.hist(data, bins=bins)

    # plt.savefig("pre2.png")  # plt.show()

    Woutarr = np.array(Wout)
    bins = 10. ** (np.arange(-8, 1, .5))
    fig = plt.figure()
    plt.xscale('log')
    plt.xlabel('Magnitude of Weights')
    plt.ylabel('Frequency of Weight')
    plt.title('Histogram of |$W_{out}$|')
    plt.hist(abs(Woutarr.flatten()), bins = bins, log = True, edgecolor='black')
    plt.savefig("pre3.png")  # plt.show()

# Full 64 bit
#np.mean(trainerre) = 0.00020750126082877475
#np.mean(tabledat,1) = array([2.08763342e-04, 1.03443018e+00, 3.99700000e+02])
#np.std(tabledat,1) = array([1.62524697e-05, 1.13443252e-01, 1.30636174e+02])
# high variance in last, from max of 595 to min of 245

# 32 bit (all squash) version
# np.mean(trainerre32) = 0.00020750152913865286
# np.mean(tabledat32,1) = array([2.08763382e-04, 1.04139317e+00, 3.99700000e+02])
# np.std(tabledat32,1) =  array([1.62520260e-05, 1.19087970e-01, 1.30636174e+02])

# 16 bit (all squash) version
# np.mean(trainerre16) = 0.0004102562615989895
# np.mean(tabledat16,1) = array([3.63208552e-04, 1.10763235e+00, 3.73700000e+02])
# np.std(tabledat16,1) =  array([1.26491718e-05, 1.13346885e-01, 1.49360001e+02])

# 32 bit (all but Wout)
# np.mean(trainerre32) = 0.00020750125226469613
# np.mean(tabledat32,1) = array([2.08763375e-04, 1.03452016e+00, 3.99700000e+02])
# np.std(tabledat32,1) =  array([1.62524058e-05, 1.38189214e-01, 1.30636174e+02])

# 16 bit (all but Wout)
# np.mean(trainerre16) = 0.0002900059516567298
# np.mean(tabledat16,1) = array([2.90876611e-04, 1.06014967e+00, 4.28200000e+02])
# np.std(tabledat16,1) =  array([1.29210463e-05, 1.48039151e-01, 1.72118448e+02])

# 32 bit (Wout only)
# np.mean(trainerre32) = 0.00020750116261588334
# np.mean(tabledat32,1) = array([2.08763349e-04, 1.04459374e+00, 3.99700000e+02])
# np.std(tabledat32,1) =  array([1.62520900e-05, 1.45641631e-01, 1.30636174e+02])

# 16 bit (Wout only)
# np.mean(trainerre16) = 0.00030781001967286706
# np.mean(tabledat16,1) = array([3.08883643e-04, 1.11597109e+00, 3.72700000e+02])
# np.std(tabledat16,1) =  array([1.42805935e-05, 8.52444048e-02, 1.48301079e+02])

# Full vs half vs 1/4 of Wout
# Full is identical to 64 bit


# 90%
# np.mean(trainerre32) = 0.00021140779528491495
# np.mean(tabledat32,1) = array([2.13702570e-04, 1.01309421e+00, 4.61800000e+02])
# np.std(tabledat32,1) =  array([1.60831195e-05, 1.87367267e-01, 2.20969138e+02])
# 85%
# np.mean(trainerre32) = 0.00021649181905168888
# np.mean(tabledat32,1) = array([2.17689377e-04, 1.10163800e+00, 3.75500000e+02])
# np.std(tabledat32,1) =  array([1.66938942e-05, 1.12351448e-01, 1.20465140e+02])
# 80%
# np.mean(trainerre16) = 0.0002242076296772851
# np.mean(tabledat16,1) = array([2.25311499e-04, 1.06687052e+00, 3.95400000e+02])
# np.std(tabledat16,1) =  array([1.65313243e-05, 1.33096337e-01, 1.99500476e+02])
# 75%
# np.mean(trainerre16) = 0.00023747827992547102
# np.mean(tabledat16,1) = array([2.39758503e-04, 1.11987629e+00, 3.36600000e+02])
# np.std(tabledat16,1) =  array([1.57498551e-05, 1.30080279e-01, 1.17350927e+02])
# 50%
# np.mean(trainerre32) = 0.000377034344395521866
# np.mean(tabledat32,1) = array([3.79906935e-04, 1.18512856e+00, 3.02500000e+02])
# np.std(tabledat32,1) =  array([1.63300750e-05, 8.14514733e-02, 1.10721497e+02])
# 25%
# np.mean(trainerre16) = 0.0005773745416764978
# np.mean(tabledat16,1) = array([5.70617552e-04, 1.14688674e+00, 2.95800000e+02])
# np.std(tabledat16,1) =  array([1.47880562e-05, 8.82466621e-02, 1.02489804e+02])


# On to ESN Version

W32prec = np.double
W16prec = np.double
# Win
Win32prec = np.single
Win16prec = np.half
# A
A32prec = np.single
A16prec = np.half
# ESN Features
ef32prec = np.single
ef16prec = np.half
# Start State
ss32prec = np.single
ss16prec = np.half
# Start Data
sd32prec = np.single
sd16prec = np.half

restrain = False
if restrain:
    x, Wout, A, Win, trainstates = train_reservoir(res_params, data[:, shift_k:shift_k + res_params['train_length']])
    print("Training Done")
    # Get training err
    trainstates2 = np.copy(trainstates)
    for j in range(2, np.shape(trainstates2)[0] - 2):
        if (np.mod(j, 2) == 0):
            trainstates2[j, :] = (trainstates[j - 1, :] * trainstates[j - 2, :]).copy()
    del trainstates

    Wout32 = np.copy(Wout).astype(W32prec)
    Wout16 = np.copy(Wout).astype(W16prec)
    Win32 = np.copy(Win).astype(Win32prec)
    Win16 = np.copy(Win).astype(Win16prec)
    A32 = np.copy(A).astype(A32prec)
    A16 = np.copy(A).astype(A16prec)




    trainout = Wout @ trainstates2
    trainerr = trainout - data[:, shift_k:shift_k + res_params['train_length']] # no need to increment by one here
    trainerre = np.linalg.norm(trainerr, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)

    trainstates2_32 = np.copy(trainstates2).astype(ef32prec)
    trainout32 = Wout32 @ trainstates2_32
    trainerr32 = trainout32 - data[:, shift_k:shift_k + res_params['train_length']]  # no need to increment by one here
    trainerre32 = np.linalg.norm(trainerr32, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)
    del trainstates2_32

    trainstates2_16 = np.copy(trainstates2).astype(ef16prec)
    trainout16 = Wout16 @ trainstates2_16
    trainerr16 = trainout16 - data[:, shift_k:shift_k + res_params['train_length']]  # no need to increment by one here
    trainerre16 = np.linalg.norm(trainerr16, axis=0) / np.linalg.norm(data[:, shift_k:shift_k + res_params['train_length']], axis=0)
    del trainstates2_16
    # Prediction

    offsets = np.array([0, 5000, 25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000])
    tabledat = np.zeros((3, len(offsets)))
    tabledat32 = np.zeros((3, len(offsets)))
    tabledat16 = np.zeros((3, len(offsets)))
    for i in range(len(offsets)):
        starttime = shift_k + res_params['train_length'] + offsets[i]
        priordat = np.copy(data[:,starttime-50:starttime])
        _, startstate = predictTF(A,Win, x*0,Wout,data[:,starttime-50:starttime])
        startstate32 = np.copy(startstate).astype(ss32prec)
        startstate16 = np.copy(startstate).astype(ss16prec)
        startdat = np.copy(data[:, starttime:starttime+res_params['predict_length']])
        startdat32 = np.copy(startdat).astype(sd32prec) # Data - should this be full?
        startdat16 = np.copy(startdat).astype(sd16prec)

        output, _ = predict(A, Win, res_params, startstate, Wout)
        output2, _ = predictTF(A, Win, startstate, Wout, startdat)
        tferr = output2 - data[:, starttime + 1:starttime + 2001]
        autoerr = output - data[:, starttime + 1:starttime + 2001]
        tferre = np.linalg.norm(tferr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        autoerre = np.linalg.norm(autoerr, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat[0, i] = np.mean(tferre)
        tabledat[1, i] = np.mean(autoerre)
        tabledat[2, i] = np.where(autoerre > .3)[0][0]  # raw number of timesteps - not divided by 200 yet
        #32 version
        output32, _ = predict(A32, Win32, res_params, startstate32, Wout32)
        output2_32, _ = predictTF(A32, Win32, startstate32, Wout32, startdat32)
        tferr32 = output2_32 - data[:, starttime + 1:starttime + 2001]
        autoerr32 = output32 - data[:, starttime + 1:starttime + 2001]
        tferre32 = np.linalg.norm(tferr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        autoerre32 = np.linalg.norm(autoerr32, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat32[0, i] = np.mean(tferre32)
        tabledat32[1, i] = np.mean(autoerre32)
        tabledat32[2, i] = np.where(autoerre32 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet
        # 16 version
        output16, _ = predict(A16, Win16, res_params, startstate16, Wout16)
        output2_16, _ = predictTF(A16, Win16, startstate16, Wout16, startdat16)
        tferr16 = output2_16 - data[:, starttime + 1:starttime + 2001]
        autoerr16 = output16 - data[:, starttime + 1:starttime + 2001]
        tferre16 = np.linalg.norm(tferr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        autoerre16 = np.linalg.norm(autoerr16, axis=0) / np.linalg.norm(data[:, starttime + 1:starttime + 2001], axis=0)
        tabledat16[0, i] = np.mean(tferre16)
        tabledat16[1, i] = np.mean(autoerre16)
        tabledat16[2, i] = np.where(autoerre16 > .3)[0][0]  # raw number of timesteps - not divided by 200 yet
        if (i == 0):  # save first set of auto predictions in each to plot
            eplot = np.copy(autoerre)
            eplot32 = np.copy(autoerre32)
            eplot16 = np.copy(autoerre16)
            sumplot = np.copy(autoerre)
        else:
            sumplot+= autoerre
    sumplot/=i
    fig = plt.figure()
    print(eplot[200:210])
    print(eplot32[200:210])
    print(eplot16[200:210])
    plt.plot(eplot)
    plt.plot(eplot32)
    plt.plot(eplot16)
    plt.legend(['64 bit', '32 bit', '16 bit'])
    plt.xlabel('Testing Step')
    plt.ylabel('Relative (Norm) Error')
    plt.title('Lorenz96 for varying precision')
    plt.xlim([0, 500])
    plt.ylim([0, 2.5])
    plt.savefig("pre4.png")  # plt.show()

# PATHAK ESN Params:
#np.mean(trainerre) = 9.646754716075445e-05
#np.mean(tabledat,1) = array([ 0.89174668,  3.70021731, 94.4       ])
#np.std(tabledat,1) = array([  0.69391108,   2.09534885, 122.17462912])
# high variance in all - ESN only approximamtes some trajectories - about half just instantly fail
# However, even taking only succesful trajectories, it still does worse than direct lorenz

# 32 bit (all squash)
#np.mean(trainerre32) = 9.639321482090696e-05
#np.mean(tabledat32,1) = array([  0.59522758,   6.85324736, 171.2       ])
#np.std(tabledat32,1) = array([  0.67643544,   6.95133653, 156.17413358])

# 16 bit (all squash)
#np.mean(trainerre16) = 0.0016702960614544127
#np.mean(tabledat16,1) = array([ 0.59539003,  7.09307693, 16.5       ])
#np.std(tabledat16,1) = array([ 0.67655076,  6.76535844, 13.90863041])

# 32 bit (all but Wout)
#np.mean(trainerre32) = 9.398552241618011e-05
#np.mean(tabledat32,1) = array([ 0.72263785,  3.66567223, 74.2       ])
#np.std(tabledat32,1) = array([  0.44496603,   1.6548098 , 130.85702121])

# 16 bit (all but Wout)
#np.mean(trainerre16) = 0.001106369448065835
#np.mean(tabledat16,1) = array([ 0.72269862,  3.74033925, 16.1       ])
#np.std(tabledat16,1) = array([ 0.44496977,  1.54235388, 25.08166661])

# 64 bit (wout only)
#np.mean(trainerre) = 9.190829101165602e-05
#np.mean(tabledat,1) = array([  0.5181812 ,   6.44196701, 204.5       ])
#np.std(tabledat,1) = array([  0.7257195 ,   7.99672333, 148.47642911])

# 32 bit (Wout only)
#np.mean(trainerre32) = 9.190840169856684e-05
#np.mean(tabledat32,1) = array([  0.51818121,   6.40655341, 204.6       ])
#np.std(tabledat32,1) = array([  0.72571951,   8.0198921 , 148.54238452])

# 16 bit (Wout only)
#np.mean(trainerre16) = 0.0010532517568276763
#np.mean(tabledat16,1) = array([ 0.51798128,  7.90588869, 28.6       ])
#np.std(tabledat16,1) = array([ 0.72540252,  7.99804424, 19.17394065])


# Devika ESN:
#np.mean(trainerre) = 0.00022166221500275062
#np.mean(tabledat,1) = array([4.27339458e-02, 1.18510333e+00, 2.61900000e+02])
#np.std(tabledat,1) = array([1.53703405e-03, 5.51220542e-02, 8.45995863e+01])
# always succeeds, but average in categories 1 and 3 (e.g. the important ones) are lower

# 32 bit (all squash)
#np.mean(trainerre32) = 0.0002140917565855414
#np.mean(tabledat32,1) = array([4.27335696e-02, 1.13505044e+00, 2.78600000e+02])
#np.std(tabledat32,1) = array([1.53918150e-03, 1.55888654e-01, 1.04472197e+02])

# 16 bit (all squash)
#np.mean(trainerre16) = 0.0007641995652762392
#np.mean(tabledat16,1) = array([4.28467989e-02, 1.26270380e+00, 1.22200000e+02])
#np.std(tabledat16,1) = array([1.54909803e-03, 9.75587612e-02, 5.18571114e+01])

# 32 bit (all but wout)
#np.mean(trainerre32) = 0.00021935392361787664
#np.mean(tabledat32,1) = array([4.27321104e-02, 1.16781981e+00, 2.46800000e+02])
#np.std(tabledat32,1) = array([1.53976207e-03, 6.31165207e-02, 6.88386519e+01])

# 16 bit (all but wout)
#np.mean(trainerre16) = 0.0005854672784582013
#np.mean(tabledat16,1) = array([4.27013376e-02, 1.27882673e+00, 1.01300000e+02])
#np.std(tabledat16,1) = array([1.53529045e-03, 8.53879045e-02, 3.29182320e+01])

# 32 bit (wout only)
#np.mean(trainerre32) = 0.0002205181846654812
#np.mean(tabledat32,1) = array([4.27308488e-02, 1.13392597e+00, 2.73200000e+02])
#np.std(tabledat32,1) = array([1.54118326e-03, 1.03835308e-01, 1.44363984e+02])

# 16 bit (wout only)
#np.mean(trainerre16) = 0.0006190603479705578
#np.mean(tabledat16,1) = array([4.28103125e-02, 1.22991153e+00, 8.89000000e+01])
#np.std(tabledat16,1) = array([1.53834835e-03, 8.76682692e-02, 2.83458992e+01])


savedat = False
if (savedat):
    import pickle
    # polyerrore
    file_polye = open('.\Pytorch\ESN\L96Polymean.pckl', 'wb')
    pickle.dump(sumplot, file_polye)
    # ESN
    file_erre = open('.\Pytorch\ESN\L96ESNmean.pckl', 'wb')
    pickle.dump(sumplot, file_erre)
    # Pathak
    file_erre2 = open('.\Pytorch\ESN\L96Pathakmean.pckl', 'wb')
    pickle.dump(sumplot, file_erre2)

# Load all data and plot
plotdat = False
if (plotdat):
    file_polye = open('.\Pytorch\ESN\L96Polymean.pckl', 'rb')
    polyerrore = pickle.load(file_polye)
    file_erre = open('.\Pytorch\ESN\L96ESNmean.pckl', 'rb')
    ESNerrore = pickle.load(file_erre)
    file_erre_p = open('.\Pytorch\ESN\L96Pathakmean.pckl', 'rb')
    Pathakerrore = pickle.load(file_erre_p)

    fig = plt.figure()
    plt.plot(polyerrore)
    plt.plot(ESNerrore)
    plt.plot(Pathakerrore)
    plt.legend(['Polynomial', 'LSR ESN', 'HSR ESN'])
    plt.xlim([0, 500])
    plt.ylim([0, 2.0])
    plt.title('Lorenz93 Performance for varying algorithms')
    plt.xlabel('Testing Step')
    plt.ylabel('Relative (Norm) Error')
    plt.savefig("pre5.png")  # plt.show()