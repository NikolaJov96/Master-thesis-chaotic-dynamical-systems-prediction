# Showing LSTM performance along an linear perturbation away from training pt during training, e.g. how performance changes across line over time

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib as mpl
from matplotlib import cm
import matplotlib.animation as animation


torch.manual_seed(1802)
num_epoch = 1250 # 10 for minibatch, 100 for full batch? - with adjusted lr, up to 800 full batch - at around 800 seems hit local min?
batch_size = 50000 # 72 for minibatch

# Set up network, data
class TakeLast(torch.nn.Module):
    def forward(self, input):
        return input[0][:,-1,:]


TrainLen = 50000
TestLen = 2000
TotLen = TrainLen + TestLen


useHenon = False
if useHenon:
    q = 4
    dim = 2  # Henon
    # Generate data - Using Henon Map
    alpha = 1.4
    b = .3
    # Xn+1 = 1 - alpha * xn**2 + yn
    # Yn+1 = b*xn
    states = np.zeros((dim, TotLen + q))
    for i in range(1, TotLen + q):
        states[:, i] = np.array(
            [1 - alpha * states[0, i - 1] * states[0, i - 1] + states[1, i - 1], b * states[0, i - 1]])

    states = np.transpose(states)/np.std(np.transpose(states), 0)
    states = np.transpose(states)

useLorenz = True
if useLorenz:
    q = 5 # 5 base
    dim = 3 # Lorenz
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    # Generate Data - Lorenz
    def f(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, (TrainLen + TestLen + q) / 200 + .005, 0.005)

    states = odeint(f, state0, t)
    regularized = np.std(states,0)
    states /= np.std(states, 0)
    states = np.transpose(states)

hsize = 50
model = torch.nn.Sequential(
    torch.nn.LSTM(dim, hidden_size = hsize, batch_first = True),
    TakeLast(),
    torch.nn.Linear(hsize,dim)
)

# Set new initial conditions of model in order to create 'unstable/spiky' start
# Output weights are mean -.005, mean abs .13, so randomly sample -.15 to .15?
# model[0].weight_hh_l0 is 200x50
# model[0].weight_ih_l0 is 200x3
# model[0].all_weights[0][0:4]
# 0 - 200x3
# 1 - 200x50
# 2 - 200 (bias?)
# 3 - 200 (bias?)
# Last two are bias_hh_l0 and bias_ih_l0
# bias_hh mean .0345, mean abs .0804
# bias_ih mean .0411, mean abs .0800
# weight_ih mean .0124, mean abs .0924
# weight_hh mean -.00570, mean abs .0886
spiky_init = False
if(spiky_init):
    with torch.no_grad():
        model[2].bias[0]-=2.5
        model[2].bias[1]-=1.5
        model[2].bias[2] -=1
        model[2].weight[:] = (torch.rand(3,50)*2-1)*5
        model[0].bias_ih_l0[:] = (torch.rand(200)*2-1)*.6 + .05
        model[0].bias_hh_l0[:] = (torch.rand(200)*2-1)*.6 + .05
        model[0].weight_ih_l0[:] = (torch.rand(200,3)*2-1)*.5
        model[0].weight_hh_l0[:] = (torch.rand(200,50)*2-1)*.4
spiky_init2 = False
if (spiky_init2):
    with torch.no_grad():
        model[2].bias[0]*=1
        model[2].bias[1]*=1
        model[2].bias[2]*=8
        model[2].weight[:]*=10
        model[0].bias_ih_l0[:]*=5
        model[0].bias_hh_l0[:]*=5
        model[0].weight_ih_l0[:]*=10
        model[0].weight_hh_l0[:]*=10
large_init = False
if large_init:
    model.load_state_dict(torch.load('./Pytorch/ESN/LSTM_Doubled.pth'))

# Lead to very wrong, but not unstable, system
# Last idea - Train to fit true data *K for k>1 e.g. 1.5, then retrain on true
# This will encourage it to be initially unstable? E.g. overshoot on every point?


# All below assumes LSTM - will need to rework if itnerested in Henon
D0 = np.array([states[0,32680],states[1,32680],states[2,32680]])
D1 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([states[0,32681],states[1,32681],states[2,32681]])
D2 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([states[0,32625],states[1,32625],states[2,32625]])
D3 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([-.5,-.75,7.5])
D1 = D1/np.linalg.norm(D1)
D2 = D2/np.linalg.norm(D2)
D3 = D3/np.linalg.norm(D3)

def testnext(model, state0, regularized):
    # Get previous true points, next true point
    state0 *= regularized
    tinv = np.arange(0.0, -q / 200, -.005)
    tnext = np.array([0,.005])
    statesinv = odeint(f, state0, tinv)
    statesinv /= regularized
    statesinv = np.transpose(statesinv)
    statenext = odeint(f, state0, tnext)
    statenext /= regularized
    statenext = np.transpose(statenext)
    state0 /= regularized # Back down to NN version for the error comparison
    # Get estimate
    input = torch.tensor(np.copy(np.flip(statesinv,1)), dtype=torch.float)
    nextpred = model(input.transpose(0, 1).view(1, q, dim))
    return nextpred.data.numpy()[0], statenext[:,1], np.sqrt(np.mean(np.square((nextpred.data.numpy() - statenext[:,1]))))


def make_LSTM_datasets(data, train_size, val_size, test_size):
    samples = train_size + val_size + test_size + q
    nfeatures = data.shape[1]
    #sdata = np.transpose(data.values)[:, :samples]
    sdata = np.transpose(data)

    Xtemp = {}
    for i in range(q):
        Xtemp[i] = sdata[:, i:samples - (q - i - 1)]

    X = Xtemp[0]
    for i in range(q - 1):
        X = np.vstack([X, Xtemp[i + 1]])

    X = np.transpose(X)
    Y = np.transpose(sdata[:, q:samples])

    Xtrain = X[:train_size, :]
    Ytrain = Y[:train_size, :]

    Xval = X[train_size:train_size + val_size, :]
    Yval = Y[train_size:train_size + val_size, :]

    Xtest = X[train_size + val_size:-1, :]
    Ytest = Y[train_size + val_size:, :]

    # reshape inputs to be 3D [samples, timesteps, features] for LSTM

    Xtrain = Xtrain.reshape((Xtrain.shape[0], q, nfeatures))
    Xval = Xval.reshape((Xval.shape[0], q, nfeatures))
    Xtest = Xtest.reshape((Xtest.shape[0], q, nfeatures))
    print("Xtrain shape = ", Xtrain.shape, "Ytrain shape = ", Ytrain.shape)
    print("Xval shape =   ", Xval.shape, "  Yval shape =   ", Yval.shape)
    print("Xtest shape =  ", Xtest.shape, " Ytest shape =  ", Ytest.shape)

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, nfeatures
Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,nfeatures = make_LSTM_datasets(np.transpose(states),TrainLen,0,TestLen)

double_target=False # do I double everything (e.g. x and y) or only y?
if (double_target):
    Ytrain*=2
    Ytest*=2

from torch.utils.data import Dataset, DataLoader
class PrepDataSet(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.tensor(np.copy(X), dtype= torch.float)
        if not torch.is_tensor(y):
            self.y = torch.tensor(np.copy(y), dtype= torch.float)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
data = PrepDataSet(X = Xtrain, y = Ytrain)
dataset = DataLoader(data, batch_size = batch_size, shuffle = False)
datatest = PrepDataSet(X = Xtest, y = Ytest)
datasettest = DataLoader(datatest, batch_size = 2000, shuffle = False)

# Get and save initial TF performance
for ix, (_xt, _yt) in enumerate(datasettest):
    y_predtest = model(_xt)
fig = plt.figure()
fig.set_size_inches(18, 10)
ax = fig.add_subplot(2,2,1,projection='3d')
ax.plot(y_predtest.data.numpy()[-TestLen:, 0], y_predtest.data.numpy()[-TestLen:, 1], y_predtest.data.numpy()[-TestLen:, 2], 'r')
ax.plot(states[0, -TestLen:], states[1, -TestLen:], states[2, -TestLen:], 'b')
ax2 = fig.add_subplot(2,2,2)
ax2.plot(y_predtest.data.numpy()[-TestLen:,0],'r')
ax2.plot(states[0,-TestLen:],'b')
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(y_predtest.data.numpy()[-TestLen:, 1], 'r')
ax3.plot(states[1, -TestLen:], 'b')
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(y_predtest.data.numpy()[-TestLen:, 2], 'r')
ax4.plot(states[2, -TestLen:], 'b')
plt.savefig('./Pytorch/ESN/Figures/LSTM_StartTF.png')

#optimizer = torch.optim.Adam(model.parameters(), lr=5e-3) # 1-2e-4 works well
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3, eps = 1e-11) # dividing by 50k fails horribly. Instead try dividing by 50? Or maybe it auto deals with the change in loss fn?
# Default eps is 1e-8, test at 1e-7 (more stable) and 1e-9 (less stable), now also 1e-11 and 1e-5
# use lr = 2e-4 for minibatch, needs higher for full batch
#loss_fn = torch.nn.MSELoss(reduction = 'sum')
loss_fn = torch.nn.MSELoss(reduction = 'mean')
losses = np.zeros(num_epoch * (TrainLen//batch_size) + 1)
losses_test = np.zeros(num_epoch * (TestLen//2000) + 1)
tostore = np.array([1010,1200,2300,2500,2700,3000,3075,4900,4940,4965,5000,5060,5400,5800,6000,8200])
tostoretest = np.array([200,300,400,500,600,700,800,900,1000]) # not handchosen, just picked 9
extend_store = np.zeros((num_epoch, 16, 3)) # each epoch, track 16 extender tasks, track 3D prediction
extend_storetest = np.zeros((num_epoch,9,3)) # each epoch, track 9 test extender tasks, 3D prediction
tfstore = np.zeros((num_epoch,2000,3))
autostore = np.zeros((num_epoch,2000,3))
testfreq = 50
D1step = np.zeros(((TrainLen//batch_size//testfreq+1)*num_epoch,100,3,3)) # Point in training sequence, Point in line sequence, summary statistic, x/y/z
kk = 0
k = 0
for e in range(num_epoch):
    if e==10  or e==25:
        for g in optimizer.param_groups:
            g['lr'] /=2
    if e==150:
        for g in optimizer.param_groups:
            g['lr'] *=2
    if e==300:
        for g in optimizer.param_groups:
            g['lr'] *=1.5
    if e==500 or e==1200:
        for g in optimizer.param_groups:
            g['lr'] /=1.5
    print('epoch '+str(e))

    for ix, (_x, _y) in enumerate(dataset): # each x/y should be a minibatch, do a standard pass using _x,_y
        # Run epsilon tube and store results
        if (np.mod(ix,testfreq)==0): # Every 100 (testfreq) batches
            for i in range(100):
                a,b,c = testnext(model, np.copy(D0 + (D1)*i/100),regularized)
                D1step[kk, i, 0, :] = a
                D1step[kk, i, 1, :] = b
                D1step[kk, i, 2, 0] = c
            kk+=1
        # Train
        y_pred = model(_x)
        loss = loss_fn(y_pred, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[ix + e*(TrainLen//batch_size)] = loss.item()
        extend_store[e,:,:] = y_pred[tostore,:].data.numpy()
    # Get Test Acc
    for ix, (_xt, _yt) in enumerate(datasettest):
        y_predtest = model(_xt)
        losstest = loss_fn(y_predtest, _yt)
        losses_test[ix + e * (TrainLen // batch_size)] = losstest.item()
        extend_storetest[e, :, :] = y_predtest[tostoretest, :].data.numpy()
        # or just store everything for his newest plot:
        tfstore[e,:,:] = y_predtest.data.numpy()
        # Unroll to get autostore, at every epoch? Might be too expensive...
        state_pred_train = np.zeros((dim, TestLen))
        i = TrainLen
        input = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
        #input[:, -1] = torch.tensor(state_pred_train[:, TrainLen - 1])
        input[:, -1] = torch.tensor(y_predtest[-1,:])
        target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)
        y_pred = model(input.transpose(0, 1).view(1, q, dim))
        state_pred_train[:, i - TrainLen] = y_pred.data.numpy()
        # Rest of iters - change iter by sliding back, overwriting last with newly predicted state_predv2
        for i in range(TrainLen + 1, TotLen):
            input[:, :-1] = input[:, 1:]
            input[:, -1] = torch.tensor(state_pred_train[:, i - TrainLen - 1])
            target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

            y_pred = model(input.transpose(0, 1).view(1, q, dim))
            state_pred_train[:, i - TrainLen] = y_pred.data.numpy()
        autostore[e,:,:]=np.transpose(state_pred_train)

#plt.plot(losses)
#plt.show()
extend_store_true_x = np.copy(_x[tostore,:,:])
extend_store_true_y = np.copy(_y[tostore,:])
extend_storetest_true_x = np.copy(_xt[tostoretest,:,:])
extend_storetest_true_y = np.copy(_yt[tostoretest,:])

# Given trained net, get training and testing performance
# Training and Teacher Forcing performance
trainloss = np.zeros(TrainLen+TestLen)
state_pred = np.zeros((dim,TrainLen+TestLen))
last_hidden = np.zeros((hsize, TrainLen + TestLen))
for i in range(TrainLen+TestLen):
    input = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    # Get and store final hidden state
    out, (h, c) = model[0](input.transpose(0, 1).view(1, q, dim))
    last_hidden[:, i] = out[0, -1, :].data.numpy()  # Final hidden state

    y_pred = model(input.transpose(0, 1).view(1, q, dim))
    loss = loss_fn(y_pred, target)
    trainloss[i] = loss.item()
    state_pred[:, i] = y_pred.data.numpy()
# Autonomous Mode
testloss = np.zeros(TestLen)
state_predv2 = np.zeros((dim,TestLen))
i = TrainLen
input = torch.tensor(np.copy(states[:, i:i+q]), dtype=torch.float)
input[:,-1] = torch.tensor(state_pred[:,TrainLen-1])
target = torch.tensor(np.copy(states[:, i+q]), dtype=torch.float)
y_pred = model(input.transpose(0, 1).view(1, q, dim))
loss = loss_fn(y_pred, target)
testloss[i-TrainLen] = loss.item()
state_predv2[:, i-TrainLen] = y_pred.data.numpy()
# Rest of iters - change iter by sliding back, overwriting last with newly predicted state_predv2
for i in range(TrainLen+1,TotLen):
    input[:,:-1] = input[:,1:]
    input[:,-1] = torch.tensor(state_predv2[:,i-TrainLen-1])
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    # Get and store final hidden state
    out, (h, c) = model[0](input.transpose(0, 1).view(1, q, dim))
    last_hidden[:, i] = out[0, -1, :].data.numpy()  # Final hidden state


    y_pred = model(input.transpose(0, 1).view(1, q, dim))
    loss = loss_fn(y_pred, target)
    testloss[i - TrainLen] = loss.item()
    state_predv2[:, i - TrainLen] = y_pred.data.numpy()


eTrain = np.transpose(Ytrain)-state_pred[:,:TrainLen]
eTrainRMSE = np.sqrt(np.mean(np.square(eTrain),0))
eTF = np.transpose(Ytest)-state_pred[:,-TestLen:]
eTFRMSE = np.sqrt(np.mean(np.square(eTF)))
eTest = np.transpose(Ytest) - state_predv2
stdTrain = np.std(Ytrain,0)
stdTF = np.std(Ytest,0)
stdTest = np.std(Ytest,0)
RRMSETrain = np.array([np.sqrt(np.mean(np.square(eTrain[:,i])/stdTrain)) for i in range(TrainLen)])
RRMSETF = np.array([np.sqrt(np.mean(np.square(eTF[:,i])/stdTF)) for i in range(TestLen)])
RRMSETest = np.array([np.sqrt(np.mean(np.square(eTest[:,i])/stdTest)) for i in range(TestLen)])

# Hyperparam tuning done. Use this as our new default LSTM implementation.
# Next major experiment is 'epsilon tube'
# Or, choose a test point, add noise (randomly in radius, and also in one of three directions) to see how 1 step ahead prediction drops compared to true next step from that perturbed point
# Find our three directions
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.view_init(30, 60)
#ax.plot(states[0, 30000:35000], states[1, 30000:35000],
#        states[2, 30000:35000], 'b')
#ax.scatter(states[0,32680],states[1,32680],states[2,32680], c = 'r')
#ax.scatter(states[0,32625],states[1,32625],states[2,32625], c = 'r')
#ax.scatter(-.5,-.7,7.5, 'r')
#ax.scatter((D0 + D1)[0],(D0 + D1)[1],(D0 + D1)[2], c = 'k')
#ax.scatter((D0 + D2)[0],(D0 + D2)[1],(D0 + D2)[2])
#ax.scatter((D0 + D3)[0],(D0 + D3)[1],(D0 + D3)[2])
#plt.draw()
#plt.show()

# Plot train error, test error
fig = plt.figure()
plt.semilogy(losses, 'b')
plt.semilogy(losses_test, 'r')
plt.title('Train and Test Losses')
plt.xlabel('Epoch')
plt.legend(['Train','Test'])
plt.savefig('./Pytorch/ESN/Figures/LSTM_Losses_tuned.png')

# stop

# Generate Animation
numplot = D1step.shape[0]
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('Testing along a line, Training Minibatches = 0')
ax = fig.add_subplot(221) # 221 if we want 4
plot = ax.semilogy(np.arange(0,1,.01),D1step[0,:,2,0])
ax.set_ylim(1e-3, 1e2)
ax2 = fig.add_subplot(222)
plot2 = ax2.plot(np.arange(0,1,.01), D1step[0,:,0,0],'r')
ax2.plot(np.arange(0,1,.01), D1step[0,:,1,0],'b')
ax2.set_ylim(-3, 1)
ax3 = fig.add_subplot(223)
plot3 = ax3.plot(np.arange(0,1,.01), D1step[0,:,0,1],'r')
ax3.plot(np.arange(0,1,.01), D1step[0,:,1,1],'b')
ax3.set_ylim(-3, 1)
ax4 = fig.add_subplot(224)
plot4 = ax4.plot(np.arange(0,1,.01), D1step[0,:,0,2],'r')
ax4.plot(np.arange(0,1,.01), D1step[0,:,1,2],'b')
ax4.set_ylim(0, 5)
def update(frame,plotin):
    fig.suptitle('Testing along a line, Training Minibatches = '+str(frame*testfreq))
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    plotin = ax.semilogy(np.arange(0,1,.01),D1step[frame,:,2,0])
    ax.set_ylim(1e-3, 1e2)
    plotin2 = ax2.plot(np.arange(0,1,.01), D1step[frame,:,0,0],'r')
    ax2.plot(np.arange(0, 1, .01), D1step[frame, :, 1, 0], 'b')
    plot3 = ax3.plot(np.arange(0, 1, .01), D1step[frame, :, 0, 1], 'r')
    ax3.plot(np.arange(0, 1, .01), D1step[frame, :, 1, 1], 'b')
    plot4 = ax4.plot(np.arange(0, 1, .01), D1step[frame, :, 0, 2], 'r')
    ax4.plot(np.arange(0,1,.01), D1step[frame,:,1,2],'b')
    ax2.set_ylim(-3, 1)
    ax3.set_ylim(-3, 1)
    ax4.set_ylim(0, 5)
    ax.set_title('RMSE')
    ax2.set_title('X Dimension')
    ax3.set_title('Y Dimension')
    ax3.set_xlabel('Distance along line')
    ax4.set_title('Z Dimension')
    ax4.set_xlabel('Distance along line')

    return plotin,
ani = animation.FuncAnimation(fig, update, numplot, fargs = (plot))
ani.save('./Pytorch/ESN/Figures/LSTM_DE.mp4',writer='ffmpeg',fps=20)


# New animation : 4x4 (16) plots of 'curve fitting task' e.g. single training examples
# Hand chosen to give a good variety (both lobes, bottom/top of lobe, inner/outer of lobe, lobe transition)
# Hand choosing - plot 3D graph (total), along with proposed segment, to find good mix
# 1010 - Left, Inner, Bottom
# 1200 - Left, Inner, Top
# 2300 - Left, Middle, Bottom
# 2500 - Left, Middle, Top
# 2700 - Transition - L to R
# 3000 - Left, Middle, Right
# 3075 - Left, Middle, Left
# 4900 - Left, Outer, Bottom/Left
# 4940 - Left, Outer, Top
# 4965 - Transition - L to R
# 5000 - Right, Inner, Bottom
# 5060 - Right, Inner, Top
# 5400 - Transition - R to L
# 5800 - Right, Middle, Right
# 6000 - Transition and or Right, Very bottom extreme
# 8200 - Right, Inner, Left
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#CandStart = 8200
#ax.plot(states[0,CandStart:CandStart+5], states[1,CandStart:CandStart+5], states[2,CandStart:CandStart+5], 'r') # Candidate
#ax.scatter(states[0,CandStart+5], states[1,CandStart+5], states[2,CandStart+5], c = 'r')
#ax.plot(states[0, -TestLen:], states[1, -TestLen:], states[2, -TestLen:], 'b') # ground truth - Example
#ax.set_xlim([states[0,CandStart+5]-1,states[0,CandStart+5]+1])
#ax.set_ylim([states[1,CandStart+5]-1,states[1,CandStart+5]+1])
#ax.set_zlim([states[2,CandStart+5]-1,states[2,CandStart+5]+1])
#plt.savefig('./Pytorch/ESN/Figures/LSTM_TestTraj.png')

# Save Data during training, create 2nd animation here
numplot = num_epoch# testing steps
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('Performance on individual Next Prediction tasks, Epoch = 0')
ax = fig.add_subplot(4,4,1, projection='3d') # 221 if we want 4
plot = ax.plot(extend_store_true_x[0,:,0],extend_store_true_x[0,:,1],extend_store_true_x[0,:,2])
ax.scatter(extend_store_true_y[0,0],extend_store_true_y[0,1],extend_store_true_y[0,2], c = 'b')
ax.scatter(extend_store[0,0,0],extend_store[0,0,1],extend_store[0,0,2],c = 'r')
ax2 = fig.add_subplot(4,4,2, projection='3d') # 221 if we want 4
plot2 = ax2.plot(extend_store_true_x[1,:,0],extend_store_true_x[1,:,1],extend_store_true_x[1,:,2])
ax2.scatter(extend_store_true_y[1,0],extend_store_true_y[1,1],extend_store_true_y[1,2], c = 'b')
ax2.scatter(extend_store[0,1,0],extend_store[0,1,1],extend_store[0,1,2],c = 'r')
ax3 = fig.add_subplot(4,4,3, projection='3d') # 221 if we want 4
plot3 = ax3.plot(extend_store_true_x[2,:,0],extend_store_true_x[2,:,1],extend_store_true_x[2,:,2])
ax3.scatter(extend_store_true_y[2,0],extend_store_true_y[2,1],extend_store_true_y[2,2], c = 'b')
ax3.scatter(extend_store[0,2,0],extend_store[0,2,1],extend_store[0,2,2],c = 'r')
ax4 = fig.add_subplot(4,4,4, projection='3d') # 221 if we want 4
plot4 = ax4.plot(extend_store_true_x[3,:,0],extend_store_true_x[3,:,1],extend_store_true_x[3,:,2])
ax4.scatter(extend_store_true_y[3,0],extend_store_true_y[3,1],extend_store_true_y[3,2], c = 'b')
ax4.scatter(extend_store[0,3,0],extend_store[0,3,1],extend_store[0,3,2],c = 'r')
ax5 = fig.add_subplot(4,4,5, projection='3d') # 221 if we want 4
plot5 = ax5.plot(extend_store_true_x[4,:,0],extend_store_true_x[4,:,1],extend_store_true_x[4,:,2])
ax5.scatter(extend_store_true_y[4,0],extend_store_true_y[4,1],extend_store_true_y[4,2], c = 'b')
ax5.scatter(extend_store[0,4,0],extend_store[0,4,1],extend_store[0,4,2],c = 'r')
ax6 = fig.add_subplot(4,4,6, projection='3d') # 221 if we want 4
plot6 = ax6.plot(extend_store_true_x[5,:,0],extend_store_true_x[5,:,1],extend_store_true_x[5,:,2])
ax6.scatter(extend_store_true_y[5,0],extend_store_true_y[5,1],extend_store_true_y[5,2], c = 'b')
ax6.scatter(extend_store[0,5,0],extend_store[0,5,1],extend_store[0,5,2],c = 'r')
ax7 = fig.add_subplot(4,4,7, projection='3d') # 221 if we want 4
plot7 = ax7.plot(extend_store_true_x[6,:,0],extend_store_true_x[6,:,1],extend_store_true_x[6,:,2])
ax7.scatter(extend_store_true_y[6,0],extend_store_true_y[6,1],extend_store_true_y[6,2], c = 'b')
ax7.scatter(extend_store[0,6,0],extend_store[0,6,1],extend_store[0,6,2],c = 'r')
ax8 = fig.add_subplot(4,4,8, projection='3d') # 221 if we want 4
plot8 = ax8.plot(extend_store_true_x[7,:,0],extend_store_true_x[7,:,1],extend_store_true_x[7,:,2])
ax8.scatter(extend_store_true_y[7,0],extend_store_true_y[7,1],extend_store_true_y[7,2], c = 'b')
ax8.scatter(extend_store[0,7,0],extend_store[0,7,1],extend_store[0,7,2],c = 'r')
ax9 = fig.add_subplot(4,4,9, projection='3d') # 221 if we want 4
plot9 = ax9.plot(extend_store_true_x[8,:,0],extend_store_true_x[8,:,1],extend_store_true_x[8,:,2])
ax9.scatter(extend_store_true_y[8,0],extend_store_true_y[8,1],extend_store_true_y[8,2], c = 'b')
ax9.scatter(extend_store[0,8,0],extend_store[0,8,1],extend_store[0,8,2],c = 'r')
ax10 = fig.add_subplot(4,4,10, projection='3d') # 221 if we want 4
plot10 = ax10.plot(extend_store_true_x[9,:,0],extend_store_true_x[9,:,1],extend_store_true_x[9,:,2])
ax10.scatter(extend_store_true_y[9,0],extend_store_true_y[9,1],extend_store_true_y[9,2], c = 'b')
ax10.scatter(extend_store[0,9,0],extend_store[0,9,1],extend_store[0,9,2],c = 'r')
ax11 = fig.add_subplot(4,4,11, projection='3d') # 221 if we want 4
plot11 = ax11.plot(extend_store_true_x[10,:,0],extend_store_true_x[10,:,1],extend_store_true_x[10,:,2])
ax11.scatter(extend_store_true_y[10,0],extend_store_true_y[10,1],extend_store_true_y[10,2], c = 'b')
ax11.scatter(extend_store[0,10,0],extend_store[0,10,1],extend_store[0,10,2],c = 'r')
ax12 = fig.add_subplot(4,4,12, projection='3d') # 221 if we want 4
plot12 = ax12.plot(extend_store_true_x[11,:,0],extend_store_true_x[11,:,1],extend_store_true_x[11,:,2])
ax12.scatter(extend_store_true_y[11,0],extend_store_true_y[11,1],extend_store_true_y[11,2], c = 'b')
ax12.scatter(extend_store[0,11,0],extend_store[0,11,1],extend_store[0,11,2],c = 'r')
ax13 = fig.add_subplot(4,4,13, projection='3d') # 221 if we want 4
plot13 = ax13.plot(extend_store_true_x[12,:,0],extend_store_true_x[12,:,1],extend_store_true_x[12,:,2])
ax13.scatter(extend_store_true_y[12,0],extend_store_true_y[12,1],extend_store_true_y[12,2], c = 'b')
ax13.scatter(extend_store[0,12,0],extend_store[0,12,1],extend_store[0,12,2],c = 'r')
ax14 = fig.add_subplot(4,4,14, projection='3d') # 221 if we want 4
plot14 = ax14.plot(extend_store_true_x[13,:,0],extend_store_true_x[13,:,1],extend_store_true_x[13,:,2])
ax14.scatter(extend_store_true_y[13,0],extend_store_true_y[13,1],extend_store_true_y[13,2], c = 'b')
ax14.scatter(extend_store[0,13,0],extend_store[0,13,1],extend_store[0,13,2],c = 'r')
ax15 = fig.add_subplot(4,4,15, projection='3d') # 221 if we want 4
plot15 = ax15.plot(extend_store_true_x[14,:,0],extend_store_true_x[14,:,1],extend_store_true_x[14,:,2])
ax15.scatter(extend_store_true_y[14,0],extend_store_true_y[14,1],extend_store_true_y[14,2], c = 'b')
ax15.scatter(extend_store[0,14,0],extend_store[0,14,1],extend_store[0,14,2],c = 'r')
ax16 = fig.add_subplot(4,4,16, projection='3d') # 221 if we want 4
plot16 = ax16.plot(extend_store_true_x[15,:,0],extend_store_true_x[15,:,1],extend_store_true_x[15,:,2])
ax16.scatter(extend_store_true_y[15,0],extend_store_true_y[15,1],extend_store_true_y[15,2], c = 'b')
ax16.scatter(extend_store[0,15,0],extend_store[0,15,1],extend_store[0,15,2],c = 'r')


anim2small = True
anim2large = False

def update2(frame,plotin):
    fig.suptitle('Performance on individual Next Prediction tasks, Epoch = '+str(frame))
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax8.clear()
    ax9.clear()
    ax10.clear()
    ax11.clear()
    ax12.clear()
    ax13.clear()
    ax14.clear()
    ax15.clear()
    ax16.clear()
    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)
    ax9.grid(True)
    ax10.grid(True)
    ax11.grid(True)
    ax12.grid(True)
    ax13.grid(True)
    ax14.grid(True)
    ax15.grid(True)
    ax16.grid(True)

    if(anim2large): # +/- 2
        ax.set_xlim(extend_store_true_y[0, 0] - 2, extend_store_true_y[0, 0] + 2)
        ax.set_ylim(extend_store_true_y[0, 1] - 2, extend_store_true_y[0, 1] + 2)
        ax.set_zlim(extend_store_true_y[0, 2] - 2, extend_store_true_y[0, 2] + 2)
        ax2.set_xlim(extend_store_true_y[1, 0] - 2, extend_store_true_y[1, 0] + 2)
        ax2.set_ylim(extend_store_true_y[1, 1] - 2, extend_store_true_y[1, 1] + 2)
        ax2.set_zlim(extend_store_true_y[1, 2] - 2, extend_store_true_y[1, 2] + 2)
        ax3.set_xlim(extend_store_true_y[2, 0] - 2, extend_store_true_y[2, 0] + 2)
        ax3.set_ylim(extend_store_true_y[2, 1] - 2, extend_store_true_y[2, 1] + 2)
        ax3.set_zlim(extend_store_true_y[2, 2] - 2, extend_store_true_y[2, 2] + 2)
        ax4.set_xlim(extend_store_true_y[3, 0] - 2, extend_store_true_y[3, 0] + 2)
        ax4.set_ylim(extend_store_true_y[3, 1] - 2, extend_store_true_y[3, 1] + 2)
        ax4.set_zlim(extend_store_true_y[3, 2] - 2, extend_store_true_y[3, 2] + 2)
        ax5.set_xlim(extend_store_true_y[4, 0] - 2, extend_store_true_y[4, 0] + 2)
        ax5.set_ylim(extend_store_true_y[4, 1] - 2, extend_store_true_y[4, 1] + 2)
        ax5.set_zlim(extend_store_true_y[4, 2] - 2, extend_store_true_y[4, 2] + 2)
        ax6.set_xlim(extend_store_true_y[5, 0] - 2, extend_store_true_y[5, 0] + 2)
        ax6.set_ylim(extend_store_true_y[5, 1] - 2, extend_store_true_y[5, 1] + 2)
        ax6.set_zlim(extend_store_true_y[5, 2] - 2, extend_store_true_y[5, 2] + 2)
        ax7.set_xlim(extend_store_true_y[6, 0] - 2, extend_store_true_y[6, 0] + 2)
        ax7.set_ylim(extend_store_true_y[6, 1] - 2, extend_store_true_y[6, 1] + 2)
        ax7.set_zlim(extend_store_true_y[6, 2] - 2, extend_store_true_y[6, 2] + 2)
        ax8.set_xlim(extend_store_true_y[7, 0] - 2, extend_store_true_y[7, 0] + 2)
        ax8.set_ylim(extend_store_true_y[7, 1] - 2, extend_store_true_y[7, 1] + 2)
        ax8.set_zlim(extend_store_true_y[7, 2] - 2, extend_store_true_y[7, 2] + 2)
        ax9.set_xlim(extend_store_true_y[8, 0] - 2, extend_store_true_y[8, 0] + 2)
        ax9.set_ylim(extend_store_true_y[8, 1] - 2, extend_store_true_y[8, 1] + 2)
        ax9.set_zlim(extend_store_true_y[8, 2] - 2, extend_store_true_y[8, 2] + 2)
        ax10.set_xlim(extend_store_true_y[9, 0] - 2, extend_store_true_y[9, 0] + 2)
        ax10.set_ylim(extend_store_true_y[9, 1] - 2, extend_store_true_y[9, 1] + 2)
        ax10.set_zlim(extend_store_true_y[9, 2] - 2, extend_store_true_y[9, 2] + 2)
        ax11.set_xlim(extend_store_true_y[10, 0] - 2, extend_store_true_y[10, 0] + 2)
        ax11.set_ylim(extend_store_true_y[10, 1] - 2, extend_store_true_y[10, 1] + 2)
        ax11.set_zlim(extend_store_true_y[10, 2] - 2, extend_store_true_y[10, 2] + 2)
        ax12.set_xlim(extend_store_true_y[11, 0] - 2, extend_store_true_y[11, 0] + 2)
        ax12.set_ylim(extend_store_true_y[11, 1] - 2, extend_store_true_y[11, 1] + 2)
        ax12.set_zlim(extend_store_true_y[11, 2] - 2, extend_store_true_y[11, 2] + 2)
        ax13.set_xlim(extend_store_true_y[12, 0] - 2, extend_store_true_y[12, 0] + 2)
        ax13.set_ylim(extend_store_true_y[12, 1] - 2, extend_store_true_y[12, 1] + 2)
        ax13.set_zlim(extend_store_true_y[12, 2] - 2, extend_store_true_y[12, 2] + 2)
        ax14.set_xlim(extend_store_true_y[13, 0] - 2, extend_store_true_y[13, 0] + 2)
        ax14.set_ylim(extend_store_true_y[13, 1] - 2, extend_store_true_y[13, 1] + 2)
        ax14.set_zlim(extend_store_true_y[13, 2] - 2, extend_store_true_y[13, 2] + 2)
        ax15.set_xlim(extend_store_true_y[14, 0] - 2, extend_store_true_y[14, 0] + 2)
        ax15.set_ylim(extend_store_true_y[14, 1] - 2, extend_store_true_y[14, 1] + 2)
        ax15.set_zlim(extend_store_true_y[14, 2] - 2, extend_store_true_y[14, 2] + 2)
        ax16.set_xlim(extend_store_true_y[15, 0] - 2, extend_store_true_y[15, 0] + 2)
        ax16.set_ylim(extend_store_true_y[15, 1] - 2, extend_store_true_y[15, 1] + 2)
        ax16.set_zlim(extend_store_true_y[15, 2] - 2, extend_store_true_y[15, 2] + 2)
    if (anim2small):  # +/- .2
        ax.set_xlim(extend_store_true_y[0, 0] - .2, extend_store_true_y[0, 0] + .2)
        ax.set_ylim(extend_store_true_y[0, 1] - .2, extend_store_true_y[0, 1] + .2)
        ax.set_zlim(extend_store_true_y[0, 2] - .2, extend_store_true_y[0, 2] + .2)
        ax2.set_xlim(extend_store_true_y[1, 0] - .2, extend_store_true_y[1, 0] + .2)
        ax2.set_ylim(extend_store_true_y[1, 1] - .2, extend_store_true_y[1, 1] + .2)
        ax2.set_zlim(extend_store_true_y[1, 2] - .2, extend_store_true_y[1, 2] + .2)
        ax3.set_xlim(extend_store_true_y[2, 0] - .2, extend_store_true_y[2, 0] + .2)
        ax3.set_ylim(extend_store_true_y[2, 1] - .2, extend_store_true_y[2, 1] + .2)
        ax3.set_zlim(extend_store_true_y[2, 2] - .2, extend_store_true_y[2, 2] + .2)
        ax4.set_xlim(extend_store_true_y[3, 0] - .2, extend_store_true_y[3, 0] + .2)
        ax4.set_ylim(extend_store_true_y[3, 1] - .2, extend_store_true_y[3, 1] + .2)
        ax4.set_zlim(extend_store_true_y[3, 2] - .2, extend_store_true_y[3, 2] + .2)
        ax5.set_xlim(extend_store_true_y[4, 0] - .2, extend_store_true_y[4, 0] + .2)
        ax5.set_ylim(extend_store_true_y[4, 1] - .2, extend_store_true_y[4, 1] + .2)
        ax5.set_zlim(extend_store_true_y[4, 2] - .2, extend_store_true_y[4, 2] + .2)
        ax6.set_xlim(extend_store_true_y[5, 0] - .2, extend_store_true_y[5, 0] + .2)
        ax6.set_ylim(extend_store_true_y[5, 1] - .2, extend_store_true_y[5, 1] + .2)
        ax6.set_zlim(extend_store_true_y[5, 2] - .2, extend_store_true_y[5, 2] + .2)
        ax7.set_xlim(extend_store_true_y[6, 0] - .2, extend_store_true_y[6, 0] + .2)
        ax7.set_ylim(extend_store_true_y[6, 1] - .2, extend_store_true_y[6, 1] + .2)
        ax7.set_zlim(extend_store_true_y[6, 2] - .2, extend_store_true_y[6, 2] + .2)
        ax8.set_xlim(extend_store_true_y[7, 0] - .2, extend_store_true_y[7, 0] + .2)
        ax8.set_ylim(extend_store_true_y[7, 1] - .2, extend_store_true_y[7, 1] + .2)
        ax8.set_zlim(extend_store_true_y[7, 2] - .2, extend_store_true_y[7, 2] + .2)
        ax9.set_xlim(extend_store_true_y[8, 0] - .2, extend_store_true_y[8, 0] + .2)
        ax9.set_ylim(extend_store_true_y[8, 1] - .2, extend_store_true_y[8, 1] + .2)
        ax9.set_zlim(extend_store_true_y[8, 2] - .2, extend_store_true_y[8, 2] + .2)
        ax10.set_xlim(extend_store_true_y[9, 0] - .2, extend_store_true_y[9, 0] + .2)
        ax10.set_ylim(extend_store_true_y[9, 1] - .2, extend_store_true_y[9, 1] + .2)
        ax10.set_zlim(extend_store_true_y[9, 2] - .2, extend_store_true_y[9, 2] + .2)
        ax11.set_xlim(extend_store_true_y[10, 0] - .2, extend_store_true_y[10, 0] + .2)
        ax11.set_ylim(extend_store_true_y[10, 1] - .2, extend_store_true_y[10, 1] + .2)
        ax11.set_zlim(extend_store_true_y[10, 2] - .2, extend_store_true_y[10, 2] + .2)
        ax12.set_xlim(extend_store_true_y[11, 0] - .2, extend_store_true_y[11, 0] + .2)
        ax12.set_ylim(extend_store_true_y[11, 1] - .2, extend_store_true_y[11, 1] + .2)
        ax12.set_zlim(extend_store_true_y[11, 2] - .2, extend_store_true_y[11, 2] + .2)
        ax13.set_xlim(extend_store_true_y[12, 0] - .2, extend_store_true_y[12, 0] + .2)
        ax13.set_ylim(extend_store_true_y[12, 1] - .2, extend_store_true_y[12, 1] + .2)
        ax13.set_zlim(extend_store_true_y[12, 2] - .2, extend_store_true_y[12, 2] + .2)
        ax14.set_xlim(extend_store_true_y[13, 0] - .2, extend_store_true_y[13, 0] + .2)
        ax14.set_ylim(extend_store_true_y[13, 1] - .2, extend_store_true_y[13, 1] + .2)
        ax14.set_zlim(extend_store_true_y[13, 2] - .2, extend_store_true_y[13, 2] + .2)
        ax15.set_xlim(extend_store_true_y[14, 0] - .2, extend_store_true_y[14, 0] + .2)
        ax15.set_ylim(extend_store_true_y[14, 1] - .2, extend_store_true_y[14, 1] + .2)
        ax15.set_zlim(extend_store_true_y[14, 2] - .2, extend_store_true_y[14, 2] + .2)
        ax16.set_xlim(extend_store_true_y[15, 0] - .2, extend_store_true_y[15, 0] + .2)
        ax16.set_ylim(extend_store_true_y[15, 1] - .2, extend_store_true_y[15, 1] + .2)
        ax16.set_zlim(extend_store_true_y[15, 2] - .2, extend_store_true_y[15, 2] + .2)

    plotin = ax.plot(extend_store_true_x[0, :, 0], extend_store_true_x[0, :, 1], extend_store_true_x[0, :, 2])
    ax.scatter(extend_store_true_y[0, 0], extend_store_true_y[0, 1], extend_store_true_y[0, 2], c='b')
    ax.scatter(extend_store[frame, 0, 0], extend_store[frame, 0, 1], extend_store[frame, 0, 2], c='r')

    plotin2 = ax2.plot(extend_store_true_x[1, :, 0], extend_store_true_x[1, :, 1], extend_store_true_x[1, :, 2])
    ax2.scatter(extend_store_true_y[1, 0], extend_store_true_y[1, 1], extend_store_true_y[1, 2], c='b')
    ax2.scatter(extend_store[frame, 1, 0], extend_store[frame, 1, 1], extend_store[frame, 1, 2], c='r')

    plotin3 = ax3.plot(extend_store_true_x[2, :, 0], extend_store_true_x[2, :, 1], extend_store_true_x[2, :, 2])
    ax3.scatter(extend_store_true_y[2, 0], extend_store_true_y[2, 1], extend_store_true_y[2, 2], c='b')
    ax3.scatter(extend_store[frame, 2, 0], extend_store[frame, 2, 1], extend_store[frame, 2, 2], c='r')

    plotin4 = ax4.plot(extend_store_true_x[3, :, 0], extend_store_true_x[3, :, 1], extend_store_true_x[3, :, 2])
    ax4.scatter(extend_store_true_y[3, 0], extend_store_true_y[3, 1], extend_store_true_y[3, 2], c='b')
    ax4.scatter(extend_store[frame, 3, 0], extend_store[frame, 3, 1], extend_store[frame, 3, 2], c='r')

    plotin5 = ax5.plot(extend_store_true_x[4, :, 0], extend_store_true_x[4, :, 1], extend_store_true_x[4, :, 2])
    ax5.scatter(extend_store_true_y[4, 0], extend_store_true_y[4, 1], extend_store_true_y[4, 2], c='b')
    ax5.scatter(extend_store[frame, 4, 0], extend_store[frame, 4, 1], extend_store[frame, 4, 2], c='r')

    plotin6 = ax6.plot(extend_store_true_x[5, :, 0], extend_store_true_x[5, :, 1], extend_store_true_x[5, :, 2])
    ax6.scatter(extend_store_true_y[5, 0], extend_store_true_y[5, 1], extend_store_true_y[5, 2], c='b')
    ax6.scatter(extend_store[frame, 5, 0], extend_store[frame, 5, 1], extend_store[frame, 5, 2], c='r')

    plotin7 = ax7.plot(extend_store_true_x[6, :, 0], extend_store_true_x[6, :, 1], extend_store_true_x[6, :, 2])
    ax7.scatter(extend_store_true_y[6, 0], extend_store_true_y[6, 1], extend_store_true_y[6, 2], c='b')
    ax7.scatter(extend_store[frame, 6, 0], extend_store[frame, 6, 1], extend_store[frame, 6, 2], c='r')

    plotin8 = ax8.plot(extend_store_true_x[7, :, 0], extend_store_true_x[7, :, 1], extend_store_true_x[7, :, 2])
    ax8.scatter(extend_store_true_y[7, 0], extend_store_true_y[7, 1], extend_store_true_y[7, 2], c='b')
    ax8.scatter(extend_store[frame, 7, 0], extend_store[frame, 7, 1], extend_store[frame, 7, 2], c='r')

    plotin9 = ax9.plot(extend_store_true_x[8, :, 0], extend_store_true_x[8, :, 1], extend_store_true_x[8, :, 2])
    ax9.scatter(extend_store_true_y[8, 0], extend_store_true_y[8, 1], extend_store_true_y[8, 2], c='b')
    ax9.scatter(extend_store[frame, 8, 0], extend_store[frame, 8, 1], extend_store[frame, 8, 2], c='r')

    plotin10 = ax10.plot(extend_store_true_x[9, :, 0], extend_store_true_x[9, :, 1], extend_store_true_x[9, :, 2])
    ax10.scatter(extend_store_true_y[9, 0], extend_store_true_y[9, 1], extend_store_true_y[9, 2], c='b')
    ax10.scatter(extend_store[frame, 9, 0], extend_store[frame, 9, 1], extend_store[frame, 9, 2], c='r')

    plotin11 = ax11.plot(extend_store_true_x[10, :, 0], extend_store_true_x[10, :, 1], extend_store_true_x[10, :, 2])
    ax11.scatter(extend_store_true_y[10, 0], extend_store_true_y[10, 1], extend_store_true_y[10, 2], c='b')
    ax11.scatter(extend_store[frame, 10, 0], extend_store[frame, 10, 1], extend_store[frame, 10, 2], c='r')

    plotin12 = ax12.plot(extend_store_true_x[11, :, 0], extend_store_true_x[11, :, 1], extend_store_true_x[11, :, 2])
    ax12.scatter(extend_store_true_y[11, 0], extend_store_true_y[11, 1], extend_store_true_y[11, 2], c='b')
    ax12.scatter(extend_store[frame, 11, 0], extend_store[frame, 11, 1], extend_store[frame, 11, 2], c='r')

    plotin13 = ax13.plot(extend_store_true_x[12, :, 0], extend_store_true_x[12, :, 1], extend_store_true_x[12, :, 2])
    ax13.scatter(extend_store_true_y[12, 0], extend_store_true_y[12, 1], extend_store_true_y[12, 2], c='b')
    ax13.scatter(extend_store[frame, 12, 0], extend_store[frame, 12, 1], extend_store[frame, 12, 2], c='r')

    plotin14 = ax14.plot(extend_store_true_x[13, :, 0], extend_store_true_x[13, :, 1], extend_store_true_x[13, :, 2])
    ax14.scatter(extend_store_true_y[13, 0], extend_store_true_y[13, 1], extend_store_true_y[13, 2], c='b')
    ax14.scatter(extend_store[frame, 13, 0], extend_store[frame, 13, 1], extend_store[frame, 13, 2], c='r')

    plotin15 = ax15.plot(extend_store_true_x[14, :, 0], extend_store_true_x[14, :, 1], extend_store_true_x[14, :, 2])
    ax15.scatter(extend_store_true_y[14, 0], extend_store_true_y[14, 1], extend_store_true_y[14, 2], c='b')
    ax15.scatter(extend_store[frame, 14, 0], extend_store[frame, 14, 1], extend_store[frame, 14, 2], c='r')

    plotin16 = ax16.plot(extend_store_true_x[15, :, 0], extend_store_true_x[15, :, 1], extend_store_true_x[15, :, 2])
    ax16.scatter(extend_store_true_y[15, 0], extend_store_true_y[15, 1], extend_store_true_y[15, 2], c='b')
    ax16.scatter(extend_store[frame, 15, 0], extend_store[frame, 15, 1], extend_store[frame, 15, 2], c='r')

    ax.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 0, :] - extend_store_true_y[0, :])))))
    ax2.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 1, :] - extend_store_true_y[1, :])))))
    ax3.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 2, :] - extend_store_true_y[2, :])))))
    ax4.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 3, :] - extend_store_true_y[3, :])))))
    ax5.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 4, :] - extend_store_true_y[4, :])))))
    ax6.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 5, :] - extend_store_true_y[5, :])))))
    ax7.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 6, :] - extend_store_true_y[6, :])))))
    ax8.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 7, :] - extend_store_true_y[7, :])))))
    ax9.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 8, :] - extend_store_true_y[8, :])))))
    ax10.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 9, :] - extend_store_true_y[9, :])))))
    ax11.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 10, :] - extend_store_true_y[10, :])))))
    ax12.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 11, :] - extend_store_true_y[11, :])))))
    ax13.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 12, :] - extend_store_true_y[12, :])))))
    ax14.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 13, :] - extend_store_true_y[13, :])))))
    ax15.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 14, :] - extend_store_true_y[14, :])))))
    ax16.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_store[frame, 15, :] - extend_store_true_y[15, :])))))

    return plotin,
ani = animation.FuncAnimation(fig, update2, numplot, fargs = (plot))
ani.save('./Pytorch/ESN/Figures/LSTM_Extenders.mp4',writer='ffmpeg',fps=20)



# New animation : 3x3 (9) plots of 'curve fitting task' e.g. single TESTING examples
# Save Data during training, create 3rd animation here
numplot = num_epoch# testing steps
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('Performance on individual Next Prediction (test) tasks, Epoch = 0')
ax = fig.add_subplot(3,3,1, projection='3d') # 221 if we want 4
plot = ax.plot(extend_storetest_true_x[0,:,0],extend_storetest_true_x[0,:,1],extend_storetest_true_x[0,:,2])
ax.scatter(extend_storetest_true_y[0,0],extend_storetest_true_y[0,1],extend_storetest_true_y[0,2], c = 'b')
ax.scatter(extend_storetest[0,0,0],extend_storetest[0,0,1],extend_storetest[0,0,2],c = 'r')
ax2 = fig.add_subplot(3,3,2, projection='3d') # 221 if we want 4
plot2 = ax2.plot(extend_storetest_true_x[1,:,0],extend_storetest_true_x[1,:,1],extend_storetest_true_x[1,:,2])
ax2.scatter(extend_storetest_true_y[1,0],extend_storetest_true_y[1,1],extend_storetest_true_y[1,2], c = 'b')
ax2.scatter(extend_storetest[0,1,0],extend_storetest[0,1,1],extend_storetest[0,1,2],c = 'r')
ax3 = fig.add_subplot(3,3,3, projection='3d') # 221 if we want 4
plot3 = ax3.plot(extend_storetest_true_x[2,:,0],extend_storetest_true_x[2,:,1],extend_storetest_true_x[2,:,2])
ax3.scatter(extend_storetest_true_y[2,0],extend_storetest_true_y[2,1],extend_storetest_true_y[2,2], c = 'b')
ax3.scatter(extend_storetest[0,2,0],extend_storetest[0,2,1],extend_storetest[0,2,2],c = 'r')
ax4 = fig.add_subplot(3,3,4, projection='3d') # 221 if we want 4
plot4 = ax4.plot(extend_storetest_true_x[3,:,0],extend_storetest_true_x[3,:,1],extend_storetest_true_x[3,:,2])
ax4.scatter(extend_storetest_true_y[3,0],extend_storetest_true_y[3,1],extend_storetest_true_y[3,2], c = 'b')
ax4.scatter(extend_storetest[0,3,0],extend_storetest[0,3,1],extend_storetest[0,3,2],c = 'r')
ax5 = fig.add_subplot(3,3,5, projection='3d') # 221 if we want 4
plot5 = ax5.plot(extend_storetest_true_x[4,:,0],extend_storetest_true_x[4,:,1],extend_storetest_true_x[4,:,2])
ax5.scatter(extend_storetest_true_y[4,0],extend_storetest_true_y[4,1],extend_storetest_true_y[4,2], c = 'b')
ax5.scatter(extend_storetest[0,4,0],extend_storetest[0,4,1],extend_storetest[0,4,2],c = 'r')
ax6 = fig.add_subplot(3,3,6, projection='3d') # 221 if we want 4
plot6 = ax6.plot(extend_storetest_true_x[5,:,0],extend_storetest_true_x[5,:,1],extend_storetest_true_x[5,:,2])
ax6.scatter(extend_storetest_true_y[5,0],extend_storetest_true_y[5,1],extend_storetest_true_y[5,2], c = 'b')
ax6.scatter(extend_storetest[0,5,0],extend_storetest[0,5,1],extend_storetest[0,5,2],c = 'r')
ax7 = fig.add_subplot(3,3,7, projection='3d') # 221 if we want 4
plot7 = ax7.plot(extend_storetest_true_x[6,:,0],extend_storetest_true_x[6,:,1],extend_storetest_true_x[6,:,2])
ax7.scatter(extend_storetest_true_y[6,0],extend_storetest_true_y[6,1],extend_storetest_true_y[6,2], c = 'b')
ax7.scatter(extend_storetest[0,6,0],extend_storetest[0,6,1],extend_storetest[0,6,2],c = 'r')
ax8 = fig.add_subplot(3,3,8, projection='3d') # 221 if we want 4
plot8 = ax8.plot(extend_storetest_true_x[7,:,0],extend_storetest_true_x[7,:,1],extend_storetest_true_x[7,:,2])
ax8.scatter(extend_storetest_true_y[7,0],extend_storetest_true_y[7,1],extend_storetest_true_y[7,2], c = 'b')
ax8.scatter(extend_storetest[0,7,0],extend_storetest[0,7,1],extend_storetest[0,7,2],c = 'r')
ax9 = fig.add_subplot(3,3,9, projection='3d') # 221 if we want 4
plot9 = ax9.plot(extend_storetest_true_x[8,:,0],extend_storetest_true_x[8,:,1],extend_storetest_true_x[8,:,2])
ax9.scatter(extend_storetest_true_y[8,0],extend_storetest_true_y[8,1],extend_storetest_true_y[8,2], c = 'b')
ax9.scatter(extend_storetest[0,8,0],extend_storetest[0,8,1],extend_storetest[0,8,2],c = 'r')


anim3small = True
anim3large = False

def update3(frame,plotin):
    fig.suptitle('Performance on individual Next Prediction tasks, Epoch = '+str(frame))
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax8.clear()
    ax9.clear()

    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    ax6.grid(True)
    ax7.grid(True)
    ax8.grid(True)
    ax9.grid(True)


    if(anim3large): # +/- 2
        ax.set_xlim(extend_storetest_true_y[0, 0] - 2, extend_storetest_true_y[0, 0] + 2)
        ax.set_ylim(extend_storetest_true_y[0, 1] - 2, extend_storetest_true_y[0, 1] + 2)
        ax.set_zlim(extend_storetest_true_y[0, 2] - 2, extend_storetest_true_y[0, 2] + 2)
        ax2.set_xlim(extend_storetest_true_y[1, 0] - 2, extend_storetest_true_y[1, 0] + 2)
        ax2.set_ylim(extend_storetest_true_y[1, 1] - 2, extend_storetest_true_y[1, 1] + 2)
        ax2.set_zlim(extend_storetest_true_y[1, 2] - 2, extend_storetest_true_y[1, 2] + 2)
        ax3.set_xlim(extend_storetest_true_y[2, 0] - 2, extend_storetest_true_y[2, 0] + 2)
        ax3.set_ylim(extend_storetest_true_y[2, 1] - 2, extend_storetest_true_y[2, 1] + 2)
        ax3.set_zlim(extend_storetest_true_y[2, 2] - 2, extend_storetest_true_y[2, 2] + 2)
        ax4.set_xlim(extend_storetest_true_y[3, 0] - 2, extend_storetest_true_y[3, 0] + 2)
        ax4.set_ylim(extend_storetest_true_y[3, 1] - 2, extend_storetest_true_y[3, 1] + 2)
        ax4.set_zlim(extend_storetest_true_y[3, 2] - 2, extend_storetest_true_y[3, 2] + 2)
        ax5.set_xlim(extend_storetest_true_y[4, 0] - 2, extend_storetest_true_y[4, 0] + 2)
        ax5.set_ylim(extend_storetest_true_y[4, 1] - 2, extend_storetest_true_y[4, 1] + 2)
        ax5.set_zlim(extend_storetest_true_y[4, 2] - 2, extend_storetest_true_y[4, 2] + 2)
        ax6.set_xlim(extend_storetest_true_y[5, 0] - 2, extend_storetest_true_y[5, 0] + 2)
        ax6.set_ylim(extend_storetest_true_y[5, 1] - 2, extend_storetest_true_y[5, 1] + 2)
        ax6.set_zlim(extend_storetest_true_y[5, 2] - 2, extend_storetest_true_y[5, 2] + 2)
        ax7.set_xlim(extend_storetest_true_y[6, 0] - 2, extend_storetest_true_y[6, 0] + 2)
        ax7.set_ylim(extend_storetest_true_y[6, 1] - 2, extend_storetest_true_y[6, 1] + 2)
        ax7.set_zlim(extend_storetest_true_y[6, 2] - 2, extend_storetest_true_y[6, 2] + 2)
        ax8.set_xlim(extend_storetest_true_y[7, 0] - 2, extend_storetest_true_y[7, 0] + 2)
        ax8.set_ylim(extend_storetest_true_y[7, 1] - 2, extend_storetest_true_y[7, 1] + 2)
        ax8.set_zlim(extend_storetest_true_y[7, 2] - 2, extend_storetest_true_y[7, 2] + 2)
        ax9.set_xlim(extend_storetest_true_y[8, 0] - 2, extend_storetest_true_y[8, 0] + 2)
        ax9.set_ylim(extend_storetest_true_y[8, 1] - 2, extend_storetest_true_y[8, 1] + 2)
        ax9.set_zlim(extend_storetest_true_y[8, 2] - 2, extend_storetest_true_y[8, 2] + 2)

    if (anim3small):  # +/- .2
        ax.set_xlim(extend_storetest_true_y[0, 0] - .2, extend_storetest_true_y[0, 0] + .2)
        ax.set_ylim(extend_storetest_true_y[0, 1] - .2, extend_storetest_true_y[0, 1] + .2)
        ax.set_zlim(extend_storetest_true_y[0, 2] - .2, extend_storetest_true_y[0, 2] + .2)
        ax2.set_xlim(extend_storetest_true_y[1, 0] - .2, extend_storetest_true_y[1, 0] + .2)
        ax2.set_ylim(extend_storetest_true_y[1, 1] - .2, extend_storetest_true_y[1, 1] + .2)
        ax2.set_zlim(extend_storetest_true_y[1, 2] - .2, extend_storetest_true_y[1, 2] + .2)
        ax3.set_xlim(extend_storetest_true_y[2, 0] - .2, extend_storetest_true_y[2, 0] + .2)
        ax3.set_ylim(extend_storetest_true_y[2, 1] - .2, extend_storetest_true_y[2, 1] + .2)
        ax3.set_zlim(extend_storetest_true_y[2, 2] - .2, extend_storetest_true_y[2, 2] + .2)
        ax4.set_xlim(extend_storetest_true_y[3, 0] - .2, extend_storetest_true_y[3, 0] + .2)
        ax4.set_ylim(extend_storetest_true_y[3, 1] - .2, extend_storetest_true_y[3, 1] + .2)
        ax4.set_zlim(extend_storetest_true_y[3, 2] - .2, extend_storetest_true_y[3, 2] + .2)
        ax5.set_xlim(extend_storetest_true_y[4, 0] - .2, extend_storetest_true_y[4, 0] + .2)
        ax5.set_ylim(extend_storetest_true_y[4, 1] - .2, extend_storetest_true_y[4, 1] + .2)
        ax5.set_zlim(extend_storetest_true_y[4, 2] - .2, extend_storetest_true_y[4, 2] + .2)
        ax6.set_xlim(extend_storetest_true_y[5, 0] - .2, extend_storetest_true_y[5, 0] + .2)
        ax6.set_ylim(extend_storetest_true_y[5, 1] - .2, extend_storetest_true_y[5, 1] + .2)
        ax6.set_zlim(extend_storetest_true_y[5, 2] - .2, extend_storetest_true_y[5, 2] + .2)
        ax7.set_xlim(extend_storetest_true_y[6, 0] - .2, extend_storetest_true_y[6, 0] + .2)
        ax7.set_ylim(extend_storetest_true_y[6, 1] - .2, extend_storetest_true_y[6, 1] + .2)
        ax7.set_zlim(extend_storetest_true_y[6, 2] - .2, extend_storetest_true_y[6, 2] + .2)
        ax8.set_xlim(extend_storetest_true_y[7, 0] - .2, extend_storetest_true_y[7, 0] + .2)
        ax8.set_ylim(extend_storetest_true_y[7, 1] - .2, extend_storetest_true_y[7, 1] + .2)
        ax8.set_zlim(extend_storetest_true_y[7, 2] - .2, extend_storetest_true_y[7, 2] + .2)
        ax9.set_xlim(extend_storetest_true_y[8, 0] - .2, extend_storetest_true_y[8, 0] + .2)
        ax9.set_ylim(extend_storetest_true_y[8, 1] - .2, extend_storetest_true_y[8, 1] + .2)
        ax9.set_zlim(extend_storetest_true_y[8, 2] - .2, extend_storetest_true_y[8, 2] + .2)

    plotin = ax.plot(extend_storetest_true_x[0, :, 0], extend_storetest_true_x[0, :, 1], extend_storetest_true_x[0, :, 2])
    ax.scatter(extend_storetest_true_y[0, 0], extend_storetest_true_y[0, 1], extend_storetest_true_y[0, 2], c='b')
    ax.scatter(extend_storetest[frame, 0, 0], extend_storetest[frame, 0, 1], extend_storetest[frame, 0, 2], c='r')

    plotin2 = ax2.plot(extend_storetest_true_x[1, :, 0], extend_storetest_true_x[1, :, 1], extend_storetest_true_x[1, :, 2])
    ax2.scatter(extend_storetest_true_y[1, 0], extend_storetest_true_y[1, 1], extend_storetest_true_y[1, 2], c='b')
    ax2.scatter(extend_storetest[frame, 1, 0], extend_storetest[frame, 1, 1], extend_storetest[frame, 1, 2], c='r')

    plotin3 = ax3.plot(extend_storetest_true_x[2, :, 0], extend_storetest_true_x[2, :, 1], extend_storetest_true_x[2, :, 2])
    ax3.scatter(extend_storetest_true_y[2, 0], extend_storetest_true_y[2, 1], extend_storetest_true_y[2, 2], c='b')
    ax3.scatter(extend_storetest[frame, 2, 0], extend_storetest[frame, 2, 1], extend_storetest[frame, 2, 2], c='r')

    plotin4 = ax4.plot(extend_storetest_true_x[3, :, 0], extend_storetest_true_x[3, :, 1], extend_storetest_true_x[3, :, 2])
    ax4.scatter(extend_storetest_true_y[3, 0], extend_storetest_true_y[3, 1], extend_storetest_true_y[3, 2], c='b')
    ax4.scatter(extend_storetest[frame, 3, 0], extend_storetest[frame, 3, 1], extend_storetest[frame, 3, 2], c='r')

    plotin5 = ax5.plot(extend_storetest_true_x[4, :, 0], extend_storetest_true_x[4, :, 1], extend_storetest_true_x[4, :, 2])
    ax5.scatter(extend_storetest_true_y[4, 0], extend_storetest_true_y[4, 1], extend_storetest_true_y[4, 2], c='b')
    ax5.scatter(extend_storetest[frame, 4, 0], extend_storetest[frame, 4, 1], extend_storetest[frame, 4, 2], c='r')

    plotin6 = ax6.plot(extend_storetest_true_x[5, :, 0], extend_storetest_true_x[5, :, 1], extend_storetest_true_x[5, :, 2])
    ax6.scatter(extend_storetest_true_y[5, 0], extend_storetest_true_y[5, 1], extend_storetest_true_y[5, 2], c='b')
    ax6.scatter(extend_storetest[frame, 5, 0], extend_storetest[frame, 5, 1], extend_storetest[frame, 5, 2], c='r')

    plotin7 = ax7.plot(extend_storetest_true_x[6, :, 0], extend_storetest_true_x[6, :, 1], extend_storetest_true_x[6, :, 2])
    ax7.scatter(extend_storetest_true_y[6, 0], extend_storetest_true_y[6, 1], extend_storetest_true_y[6, 2], c='b')
    ax7.scatter(extend_storetest[frame, 6, 0], extend_storetest[frame, 6, 1], extend_storetest[frame, 6, 2], c='r')

    plotin8 = ax8.plot(extend_storetest_true_x[7, :, 0], extend_storetest_true_x[7, :, 1], extend_storetest_true_x[7, :, 2])
    ax8.scatter(extend_storetest_true_y[7, 0], extend_storetest_true_y[7, 1], extend_storetest_true_y[7, 2], c='b')
    ax8.scatter(extend_storetest[frame, 7, 0], extend_storetest[frame, 7, 1], extend_storetest[frame, 7, 2], c='r')

    plotin9 = ax9.plot(extend_storetest_true_x[8, :, 0], extend_storetest_true_x[8, :, 1], extend_storetest_true_x[8, :, 2])
    ax9.scatter(extend_storetest_true_y[8, 0], extend_storetest_true_y[8, 1], extend_storetest_true_y[8, 2], c='b')
    ax9.scatter(extend_storetest[frame, 8, 0], extend_storetest[frame, 8, 1], extend_storetest[frame, 8, 2], c='r')

    ax.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 0, :] - extend_storetest_true_y[0, :])))))
    ax2.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 1, :] - extend_storetest_true_y[1, :])))))
    ax3.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 2, :] - extend_storetest_true_y[2, :])))))
    ax4.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 3, :] - extend_storetest_true_y[3, :])))))
    ax5.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 4, :] - extend_storetest_true_y[4, :])))))
    ax6.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 5, :] - extend_storetest_true_y[5, :])))))
    ax7.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 6, :] - extend_storetest_true_y[6, :])))))
    ax8.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 7, :] - extend_storetest_true_y[7, :])))))
    ax9.set_title("RMSE: {:.9f}".format(np.sqrt(np.mean(np.square(extend_storetest[frame, 8, :] - extend_storetest_true_y[8, :])))))

    return plotin,
ani = animation.FuncAnimation(fig, update3, numplot, fargs = (plot))
ani.save('./Pytorch/ESN/Figures/LSTM_TestExtenders.mp4',writer='ffmpeg',fps=20)

# Get and save equivelant plots from the LSTM_Hyper - e.g. directional and random errors. Compare to ESN
D1steperror = np.zeros(1000)
D2steperror = np.zeros(1000)
D3steperror = np.zeros(1000)
for i in range(1000):
    a,b,c = testnext(model, np.copy(D0 + (D1)*i/1000),regularized)
    D1steperror[i] = c
    a, b, c = testnext(model, np.copy(D0 + (D2) * i / 1000), regularized)
    D2steperror[i] = c
    a, b, c = testnext(model, np.copy(D0 + (D3) * i / 1000), regularized)
    D3steperror[i] = c
fig = plt.figure()
plt.plot(np.arange(0,1,.001),D1steperror)
plt.plot(np.arange(0,1,.001),D2steperror)
plt.plot(np.arange(0,1,.001),D3steperror)
plt.legend(['D1', 'D2', 'D3 - Outside Attractor'])
plt.title('LSTM - One Step Evolution Relative Errors, Fixed Perturbations')
plt.savefig('./Pytorch/ESN/Figures/LSTM_DEps_tuned.png')
# Random direction of fixed epsilon
Esteperrors = np.zeros((100,2))
for i in range(100): #.01 to 1 distance
    Ehold = np.zeros(25)
    for j in range(25): # 25 samples each
        DE = np.random.rand(3)*2 - 1 #-1 to 1 uniform
        DE = DE/np.linalg.norm(DE) * (i+1)/100
        a,b,c = testnext(model, np.copy(D0 + DE), regularized)
        Ehold[j] = c
    Esteperrors[i,0] = np.mean(Ehold)
    Esteperrors[i,1] = np.std(Ehold)
fig = plt.figure()
plt.errorbar(np.arange(.01,1.01,.01), Esteperrors[:,0], yerr=Esteperrors[:,1], label='Random Perturbations')
plt.title('LSTM - One Step Evolution Relative Errors, Random Perturbation')
plt.savefig('./Pytorch/ESN/Figures/LSTM_RandEps_tuned.png')

# View Prediction
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(state_pred[0,-TestLen:], state_pred[1,-TestLen:], state_pred[2,-TestLen:],'r')
ax.plot(states[0,-TestLen:],states[1,-TestLen:],states[2,-TestLen:],'b')
plt.savefig('./Pytorch/ESN/Figures/LSTM_TFPred_tuned.png')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(state_predv2[0,:], state_predv2[1,:], state_predv2[2,:], 'r')
ax.plot(states[0, -TestLen:], states[1, -TestLen:], states[2, -TestLen:], 'b')
plt.savefig('./Pytorch/ESN/Figures/LSTM_AutoPred_tuned.png')

#fig = plt.figure()
#plt.plot(state_predv2[2,:])
#plt.savefig('./Pytorch/ESN/Figures/Test.png')

# Check LSTM errors (at fixed time) for bias. I expect them to be 0 biased overall, but confirm that first


# Display eigs of update eqn? Don't have the 'true' derivatives (and can't use autograd to compare, as hidden states are indpendent in the graph)
# But we can estimate it as h_q_t - h_q_(t-1) : However, we need the Jacobian, not merely the temporal derivative...
def get_jacobian(net, x, noutputs):
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.eye(noutputs))
    return x.grad.data
# So instead, try the above code found from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
# Given PREVIOUS hidden state (e.g. h_(q-1)), extract from the network the transform h_(q-1) to h_q - ?? This is the LSTM hidden layer update term, neglecting inputs?
# Then, given the function (LSMT), previous state (x) and number of outputs(hsize), above should give jacobian! But only for h_(q-1) to h_q...
# as a test, try it with current 'out', out [0,-1,:] is last state, [0,-2,:] is previous e.g. our x
# But how to get this? model[0].weight_hh_l0 gives us a 200x50 vector...
# Could instead just use the LSTM with a given hidden state, but this neglects the cell? Try it anyways...
model[0](input[:,-1].view(1,1,3), (out[0,-2,:].view(1,1,50),(out[0,-2,:]*0).view(1,1,50)))[0] # does not match...
# Presumably using the wrong c0 messes it up. We could potentially extract the 'true' c0 to use
out0, (h0, c0) = model[0](input[:,:-1].transpose(0, 1).view(1, q-1, dim))
model[0](input[:,-1].view(1,1,3), (h0,c0))[0] # now matches
# The fact that we require additional inputs means we can't use the above code as is
# Instead, try to redo by hand?
n = 50
testh0 = h0.squeeze()
testh0 = testh0.repeat(50,1)
testh0.requires_grad_(True)
testin = input[:,-1].view(1,1,3)
testin.requires_grad = True
testy = model[0](testin,(h0,c0))[0][0,0,0]
optimizer.zero_grad()
testy.backward(torch.eye(50),retain_graph = True)
# h0.grad.data # fails...
# Seems to only compute gradient for the inputs, not with respect to hidden state, so useless to us
# That said, can probably explicitly compute (no autograd unforunately) the ESN one, since hidden states ARE in the comp graph
# What about instead, doing eigenvalues direclty on output e.g. d(xnext)/dx, rather than on hidden? LSTM autograd may work here
# Model output is 1x1x50 - multiply the 50 by the 50x3 final output to get total output?
Jtest = np.zeros((3,3))
testx1 = torch.matmul(model[0](testin,(h0,c0))[0][-1,-1,:],model[2].weight.t())[0] # first output
# optimizer.zero_grad()
testx1.backward(torch.eye(3),retain_graph = True)
# testin.grad # exists! This should now be the first row of the jacobian - repeat for rows 2 and 3
Jtest[0,:] = testin.grad.data.numpy()[0,0,:]
testx2 = torch.matmul(model[0](testin,(h0,c0))[0][-1,-1,:],model[2].weight.t())[1] # 2nd output
optimizer.zero_grad()
testx2.backward(torch.eye(3),retain_graph = True)
Jtest[1,:] = testin.grad.data.numpy()[0,0,:]
testx3 = torch.matmul(model[0](testin,(h0,c0))[0][-1,-1,:],model[2].weight.t())[2] # 3rd output
optimizer.zero_grad()
testx3.backward(torch.eye(3),retain_graph = True)
Jtest[2,:] = testin.grad.data.numpy()[0,0,:]
# Is this enough? This is the eigenvalues of the ONE step ahead prediction, given the hidden state for q>1
# Should this be done on TF or auto?

# Generalize this to full TF trajectory
A1D1 = np.array([states[0,32470]-states[0,32680],states[1,32470]-states[1,32680],states[2,32470]-states[2,32680]])
A1D2 = np.array([states[0,32100]-states[0,32510],states[1,32100]-states[1,32510],states[2,32100]-states[2,32510]])
A2D1 = np.array([states[0,30000]-states[0,30080],states[1,30000]-states[1,30080],states[2,30000]-states[2,30080]])
A2D2 = np.array([states[0,30052]-states[0,30102],states[1,30052]-states[1,30102],states[2,30052]-states[2,30102]])
A1D1 = A1D1/np.linalg.norm(A1D1)
A1D2 = A1D2/np.linalg.norm(A1D2)
A2D1 = A2D1/np.linalg.norm(A2D1)
A2D2 = A2D2/np.linalg.norm(A2D2)


JacTF = np.zeros((TestLen,3,3))
Jaceigvals = np.zeros((TestLen,3,2)) # real/imag
Jaceigvecs = np.zeros((TestLen,3,3))
Loreigvals = np.zeros((TestLen,3,2))
Loreigvecs = np.zeros((TestLen,3,3))
for i in range(TrainLen,TrainLen+TestLen):
    input = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    out0, (h0, c0) = model[0](input[:, :-1].transpose(0, 1).view(1, q - 1, dim))
    testin = input[:, -1].view(1, 1, 3)
    testin.requires_grad = True
    Jtest = np.zeros((3, 3))
    testx1 = torch.matmul(model[0](testin, (h0, c0))[0][-1, -1, :], model[2].weight.t())[0]  # first output
    optimizer.zero_grad()
    testx1.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    Jtest[0, :] = testin.grad.data.numpy()[0, 0, :]
    testx2 = torch.matmul(model[0](testin, (h0, c0))[0][-1, -1, :], model[2].weight.t())[1]  # 2nd output
    optimizer.zero_grad()
    testx2.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    Jtest[1, :] = testin.grad.data.numpy()[0, 0, :]
    testx3 = torch.matmul(model[0](testin, (h0, c0))[0][-1, -1, :], model[2].weight.t())[2]  # 3rd output
    optimizer.zero_grad()
    testx3.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    Jtest[2, :] = testin.grad.data.numpy()[0, 0, :]

    # Store Jacobian
    JacTF[i-TrainLen,:,:] = Jtest-np.eye(3)
    # Get and store eigvals, eigvecs
    Jeigvals, Jeigvecs = np.linalg.eig(Jtest-np.eye(3))
    Jaceigvals[i-TrainLen,:,0] = np.real(Jeigvals)
    Jaceigvals[i-TrainLen,:,1] = np.imag(Jeigvals)
    Jaceigvecs[i-TrainLen,:,:] = Jeigvecs

    # Get and store true (Lorenz) Jacobean, eigenval, eigenvecs
    Jl = np.zeros((3,3)) # Eigenvalues of lorenz system
    sl = states[:,-2000+i]*regularized
    Jl[0,0] = -sigma
    Jl[0,1] = sigma
    Jl[0,2] = 0
    Jl[1,0] = rho - sl[2]
    Jl[1,1] = -1
    Jl[1,2] = -sl[0]
    Jl[2,0] = sl[1]
    Jl[2,1] = sl[0]
    Jl[2,2] = -beta
    Jl[0,:] /= regularized[0]
    Jl[1,:] /= regularized[1]
    Jl[2,:] /= regularized[2]
    Jl*=.25 #? May have to be rescaled
    #Jl+=np.eye(3)
    leigvals, leigvecs = np.linalg.eig(Jl)
    Loreigvals[i-TrainLen,:,0] = np.real(leigvals)
    Loreigvals[i-TrainLen,:,1] = np.imag(leigvals)
    Loreigvecs[i-TrainLen,:,:] = leigvecs

numplot = TestLen
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('Lorenz TF Eigenvalues, Test step = 0')
ax = fig.add_subplot(221) # 221 if we want 4
plot = ax.scatter(Jaceigvals[0,:,0], Jaceigvals[0,:,1])
ax2 = fig.add_subplot(222, projection='3d')
plot2 = ax2.scatter(state_pred[0, -2000], state_pred[1, -2000], state_pred[2, -2000], c ='r')
ax2.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
def update4(frame,plotin, Jaceigvals):
    fig.suptitle('Lorenz TF Eigenvalues, Test step = '+str(frame))
    ax.clear()
    ax2.clear()
    plotin = ax.scatter(Jaceigvals[frame,:,0], Jaceigvals[frame,:,1], c = 'b')
    ax.scatter(Loreigvals[frame,:,0],Loreigvals[frame,:,1], c = 'r')
    ax.set_ylim(-.5,.5)
    ax.set_xlim(-1,.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend(["LSTM Eigvals","Lorenz Eigvals"])
    plotin2 = ax2.scatter(state_pred[0, -2000+frame], state_pred[1, -2000+frame], state_pred[2, -2000+frame], c ='r')
    ax2.plot(states[0, -2000:], states[1, -2000:], states[2, -2000:], 'b')
    ax2.scatter(states[0, -2000+frame], states[1, -2000+frame], states[2, -2000+frame], c ='b')
    # Idea - show errors e.g. true + error*large_val in order to show if there is obvious visual pattern to errors
    # eTF = (3x200) target - prediction
    em = 250
    ax2.plot(np.array([states[0,-2000+frame], states[0,-2000+frame] + em*-eTF[0,frame]]),np.array([states[1,-2000+frame],states[1,-2000+frame] + em*-eTF[1,frame]]),np.array([states[2,-2000+frame],states[2,-2000+frame] + em*-eTF[2,frame]]), 'r')
    ax2.set_xlim(-3,3)
    ax2.set_ylim(-3,3)
    ax2.set_zlim(0,6)
    return plotin,
ani = animation.FuncAnimation(fig, update4, numplot, fargs = (plot, Jaceigvals))
ani.save('./Pytorch/ESN/Figures/LSTM_TFEig.mp4',writer='ffmpeg',fps=20)

np.mean(np.log(np.max(Jaceigvals[:,:,0]+1,1))) #-.0458
np.mean(np.log(np.max(Jaceigvals[:1000,:,0]+1,1))) #-.173 first 1k

# Two new plots - During each epoch of training, store both LSTM auto and TF trajectories to plot (animate)
numplot = num_epoch
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('LSTM Training TF, Test step = 0')
ax = fig.add_subplot(221, projection='3d') # 221 if we want 4
plot = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
ax.plot(tfstore[0,:,0],tfstore[0,:,1],tfstore[0,:,2],'r')
ax2 = fig.add_subplot(222)
plot2 = ax2.plot(states[0,-2000:],'b')
ax2.plot(tfstore[0,:,0],'r')
ax3 = fig.add_subplot(223)
plot3 = ax3.plot(states[1,-2000:],'b')
ax3.plot(tfstore[0,:,1],'r')
ax4 = fig.add_subplot(224)
plot4 = ax4.plot(states[2,-2000:],'b')
ax4.plot(tfstore[0,:,2],'r')
def update5(frame,plotin, states):
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    fig.suptitle('LSTM Training TF, Test step = ' + str(frame))
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(0,6)
    plotin = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
    ax.plot(tfstore[frame, :, 0], tfstore[frame, :, 1], tfstore[frame, :, 2], 'r')
    plot2 = ax2.plot(states[0, -2000:], 'b')
    ax2.plot(tfstore[frame, :, 0], 'r')
    plot3 = ax3.plot(states[1, -2000:], 'b')
    ax3.plot(tfstore[frame, :, 1], 'r')
    plot4 = ax4.plot(states[2, -2000:], 'b')
    ax4.plot(tfstore[frame, :, 2], 'r')
    return plotin
ani = animation.FuncAnimation(fig, update5, numplot, fargs=(plot, states))
ani.save('./Pytorch/ESN/Figures/LSTM_TrainTF.mp4', writer='ffmpeg', fps=20)
# Same thing, but for auto
numplot = num_epoch
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('LSTM Training Auto, Test step = 0')
ax = fig.add_subplot(221, projection='3d') # 221 if we want 4
plot = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
ax.plot(autostore[0,:,0],autostore[0,:,1],autostore[0,:,2],'r')
ax2 = fig.add_subplot(222)
plot2 = ax2.plot(states[0,-2000:],'b')
ax2.plot(autostore[0,:,0],'r')
ax3 = fig.add_subplot(223)
plot3 = ax3.plot(states[1,-2000:],'b')
ax3.plot(autostore[0,:,1],'r')
ax4 = fig.add_subplot(224)
plot4 = ax4.plot(states[2,-2000:],'b')
ax4.plot(autostore[0,:,2],'r')
def update6(frame,plotin, states):
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    #ax.set_xlim(-3,3)
    #ax.set_ylim(-3,3)
    #ax.set_zlim(0,6)
    fig.suptitle('LSTM Training Auto, Test step = '+str(frame))
    plotin = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
    ax.plot(autostore[frame, :, 0], autostore[frame, :, 1], autostore[frame, :, 2], 'r')
    plot2 = ax2.plot(states[0, -2000:], 'b')
    ax2.plot(autostore[frame, :, 0], 'r')
    plot3 = ax3.plot(states[1, -2000:], 'b')
    ax3.plot(autostore[frame, :, 1], 'r')
    plot4 = ax4.plot(states[2, -2000:], 'b')
    ax4.plot(autostore[frame, :, 2], 'r')
    return plotin
ani = animation.FuncAnimation(fig, update6, numplot, fargs=(plot, states))
ani.save('./Pytorch/ESN/Figures/LSTM_TrainAuto.mp4', writer='ffmpeg', fps=20)

if (False): # save/load 'doubled' model
    torch.save(model.state_dict(), './Pytorch/ESN/LSTM_Doubled.pth')
    torch.load_state_dict(torch.load('./Pytorch/ESN/LSTM_Doubled.pth'))

# New plot (animation) of ERROR over learning time, in TF Mode
numplot = num_epoch
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('LSTM Training TF Error, Epoch = 0, Train Loss = '+str(losses[0]))
ax = fig.add_subplot(221, projection='3d') # 221 if we want 4
plot = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
#axe = ax.twinx()
#axe.plot(tfstore[0,:,0]-states[0,-2000:],tfstore[0,:,1]-states[1,-2000:],tfstore[0,:,2]-states[2,-2000:],'r')
ax2 = fig.add_subplot(222)
plot2 = ax2.plot(states[0,-2000:],'b')
ax2e = ax2.twinx()
ax2e.plot(tfstore[0,:,0] - states[0,-2000:],'r')
ax3 = fig.add_subplot(223)
plot3 = ax3.plot(states[1,-2000:],'b')
ax3e = ax3.twinx()
ax3e.plot(tfstore[0,:,1] - states[1,-2000:],'r')
ax4 = fig.add_subplot(224)
plot4 = ax4.plot(states[2,-2000:],'b')
ax4e = ax4.twinx()
ax4e.plot(tfstore[0,:,2] - states[2,-2000:],'r')
def update7(frame,plotin, states):
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax2e.clear()
    ax3e.clear()
    ax4e.clear()
    fig.suptitle('LSTM Training TF Error, Epoch = ' + str(frame)+', Train Loss = '+str(losses[frame]))
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(0,6)
    plotin = ax.plot(states[0,-2000:], states[1,-2000:], states[2,-2000:], 'b')
    ax.plot(tfstore[frame, :, 0], tfstore[frame, :, 1], tfstore[frame, :, 2], 'r')
    plot2 = ax2.plot(states[0, -2000:], 'b')
    ax2e.plot(tfstore[frame, :, 0] - states[0,-2000:], 'r')
    plot3 = ax3.plot(states[1, -2000:], 'b')
    ax3e.plot(tfstore[frame, :, 1] - states[1,-2000:], 'r')
    plot4 = ax4.plot(states[2, -2000:], 'b')
    ax4e.plot(tfstore[frame, :, 2] - states[2,-2000:], 'r')
    ax2.set_title('X Output and Error')
    ax3.set_title('Y Output and Error')
    ax4.set_title('Z Output and Error')
    ax2e.set_ylim([np.min(tfstore[-1, :, 0]-states[0,-2000:]), np.max(tfstore[-1, :, 0]-states[0,-2000:])])
    ax3e.set_ylim([np.min(tfstore[-1, :, 1]-states[1,-2000:]), np.max(tfstore[-1, :, 1]-states[1,-2000:])])
    ax4e.set_ylim([np.min(tfstore[-1, :, 2]-states[2,-2000:]), np.max(tfstore[-1, :, 2]-states[2,-2000:])])
    return plotin
ani = animation.FuncAnimation(fig, update7, numplot, fargs=(plot, states))
ani.save('./Pytorch/ESN/Figures/LSTM_TrainTFError.mp4', writer='ffmpeg', fps=20)
# New plot of final errors in TF mode - scatter plot, histograms
# Want histogram of x/y/z errors (marginals), and 3D scatter plot of total error
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('LSTM Final TF Error')
ax = fig.add_subplot(221, projection='3d') # 221 if we want 4
ax.scatter(tfstore[-1,:,0] - states[0,-2000:],tfstore[-1,:,1] - states[1,-2000:],tfstore[-1,:,2] - states[2,-2000:])
ax2 = fig.add_subplot(2,2,2)
ax2.hist(tfstore[-1,:,0] - states[0,-2000:])
ax3 = fig.add_subplot(2,2,3)
ax3.hist(tfstore[-1,:,1] - states[1,-2000:])
ax4 = fig.add_subplot(2,2,4)
ax4.hist(tfstore[-1,:,2] - states[2,-2000:])
ax.set_title('Scatterplot of Errors')
ax2.set_title('X Errors')
ax3.set_title('Y Errors')
ax4.set_title('Z Errors')
plt.savefig('./Pytorch/ESN/Figures/LSTM_TrainTFErrorHist.png')


"""

/home/nx3/.conda/envs/keras/lib/python3.7/site-packages/torch/nn/modules/loss.py:432: UserWarning: Using a target size (torch.Size([3])) that is different to the input size (torch.Size([1, 3])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  return F.mse_loss(input, target, reduction=self.reduction)
Traceback (most recent call last):
  File "LSTM_EpsAnim.py", line 1045, in <module>
    Jtest[0,:] = testin.grad.data.numpy()[0,0,:]
AttributeError: 'NoneType' object has no attribute 'data'

"""