import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib.animation as animation
from torch.utils.data import Dataset, DataLoader


# Set up network, data
class TakeLast(torch.nn.Module):
    def forward(self, input):
        return input[0][:, -1, :]


torch.manual_seed(1802)
num_epoch = 1250  # 10 for minibatch, 100 for full batch? - with adjusted lr, up to 800 full batch
batch_size = 50000  # 72 for mini-batch

TrainLen = 50000
TestLen = 2000
TotLen = TrainLen + TestLen

q = 5  # 5 base
dim = 3  # Lorenz
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

h_size = 50
model = torch.nn.Sequential(
    torch.nn.LSTM(dim, hidden_size=h_size, batch_first=True),
    TakeLast(),
    torch.nn.Linear(h_size, dim)
)

# All below assumes LSTM - will need to rework if itnerested in Henon
D0 = np.array([states[0,32680],states[1,32680],states[2,32680]])
D1 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([states[0,32681],states[1,32681],states[2,32681]])
D2 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([states[0,32625],states[1,32625],states[2,32625]])
D3 = np.array([states[0,32680],states[1,32680],states[2,32680]]) - np.array([-.5,-.75,7.5])
D1 = D1/np.linalg.norm(D1)
D2 = D2/np.linalg.norm(D2)
D3 = D3/np.linalg.norm(D3)


def test_next(model, state0, regularized):
    # Get previous true points, next true point
    state0 *= regularized
    t_inv = np.arange(0.0, -q / 200, -.005)
    t_next = np.array([0, .005])
    states_inv = odeint(f, state0, t_inv)
    states_inv /= regularized
    states_inv = np.transpose(states_inv)
    state_next = odeint(f, state0, t_next)
    state_next /= regularized
    state_next = np.transpose(state_next)
    state0 /= regularized  # Back down to NN version for the error comparison
    # Get estimate
    torch_input = torch.tensor(np.copy(np.flip(states_inv, 1)), dtype=torch.float)
    next_pred = model(torch_input.transpose(0, 1).view(1, q, dim))
    return next_pred.data.numpy()[0], state_next[:, 1], np.sqrt(np.mean(np.square((next_pred.data.numpy() - state_next[:, 1]))))


def make_lstm_data_sets(data, train_size, val_size, test_size):
    samples = train_size + val_size + test_size + q
    n_features_loc = data.shape[1]
    s_data = np.transpose(data)

    x_temp = {}
    for i in range(q):
        x_temp[i] = s_data[:, i:samples - (q - i - 1)]

    x = x_temp[0]
    for i in range(q - 1):
        x = np.vstack([x, x_temp[i + 1]])

    x = np.transpose(x)
    y = np.transpose(s_data[:, q:samples])

    x_train = x[:train_size, :]
    y_train = y[:train_size, :]

    x_val = x[train_size:train_size + val_size, :]
    y_val = y[train_size:train_size + val_size, :]

    x_test = x[train_size + val_size:-1, :]
    y_test = y[train_size + val_size:, :]

    # reshape inputs to be 3D [samples, timesteps, features] for LSTM

    x_train = x_train.reshape((x_train.shape[0], q, n_features_loc))
    x_val = x_val.reshape((x_val.shape[0], q, n_features_loc))
    x_test = x_test.reshape((x_test.shape[0], q, n_features_loc))
    print("Xtrain shape = ", x_train.shape, "Ytrain shape = ", y_train.shape)
    print("Xval shape =   ", x_val.shape, "  Yval shape =   ", y_val.shape)
    print("Xtest shape =  ", x_test.shape, " Ytest shape =  ", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, n_features_loc


X_train, Y_train, X_val, Y_val, X_test, Y_test, n_features = \
    make_lstm_data_sets(np.transpose(states), TrainLen, 0, TestLen)


class PrepDataSet(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.tensor(np.copy(X), dtype=torch.float)
        if not torch.is_tensor(y):
            self.y = torch.tensor(np.copy(y), dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


data = PrepDataSet(X=X_train, y=Y_train)
dataset = DataLoader(data, batch_size=batch_size, shuffle=False)
data_test = PrepDataSet(X=X_test, y=Y_test)
data_set_test = DataLoader(data_test, batch_size=2000, shuffle=False)

# dividing by 50k fails horribly. Instead try dividing by 50? Or maybe it auto deals with the change in loss fn?
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3, eps = 1e-11)
loss_fn = torch.nn.MSELoss(reduction='mean')
losses = np.zeros(num_epoch * (TrainLen//batch_size) + 1)
losses_test = np.zeros(num_epoch * (TestLen//2000) + 1)
to_store = np.array([1010, 1200, 2300, 2500, 2700, 3000, 3075, 4900, 4940, 4965, 5000, 5060, 5400, 5800, 6000, 8200])
to_store_test = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000]) # not handchosen, just picked 9
extend_store = np.zeros((num_epoch, 16, 3)) # each epoch, track 16 extender tasks, track 3D prediction
extend_store_test = np.zeros((num_epoch, 9, 3)) # each epoch, track 9 test extender tasks, 3D prediction
tf_store = np.zeros((num_epoch, 2000, 3))
auto_store = np.zeros((num_epoch, 2000, 3))
test_freq = 50
# Point in training sequence, Point in line sequence, summary statistic, x/y/z
D1step = np.zeros(((TrainLen // batch_size // test_freq + 1) * num_epoch, 100, 3, 3))
kk = 0
k = 0
for e in range(num_epoch):
    if e == 10  or e == 25:
        for g in optimizer.param_groups:
            g['lr'] /= 2
    if e == 150:
        for g in optimizer.param_groups:
            g['lr'] *= 2
    if e == 300:
        for g in optimizer.param_groups:
            g['lr'] *= 1.5
    if e == 500 or e == 1200:
        for g in optimizer.param_groups:
            g['lr'] /= 1.5
    print('epoch ' + str(e))

    for ix, (_x, _y) in enumerate(dataset): # each x/y should be a minibatch, do a standard pass using _x,_y
        # Run epsilon tube and store results
        if np.mod(ix, test_freq) == 0:  # Every 100 (testfreq) batches
            for i in range(100):
                a, b, c = test_next(model, np.copy(D0 + (D1) * i / 100), regularized)
                D1step[kk, i, 0, :] = a
                D1step[kk, i, 1, :] = b
                D1step[kk, i, 2, 0] = c
            kk += 1
        # Train
        y_pred = model(_x)
        loss = loss_fn(y_pred, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[ix + e*(TrainLen//batch_size)] = loss.item()
        extend_store[e,:,:] = y_pred[to_store, :].data.numpy()
    # Get Test Acc
    for ix, (_xt, _yt) in enumerate(data_set_test):
        y_pred_test = model(_xt)
        loss_test = loss_fn(y_pred_test, _yt)
        losses_test[ix + e * (TrainLen // batch_size)] = loss_test.item()
        extend_store_test[e, :, :] = y_pred_test[to_store_test, :].data.numpy()
        # or just store everything for his newest plot:
        tf_store[e, :, :] = y_pred_test.data.numpy()
        # Unroll to get auto_store, at every epoch? Might be too expensive...
        state_pred_train = np.zeros((dim, TestLen))
        i = TrainLen
        input_values = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
        input_values[:, -1] = torch.tensor(y_pred_test[-1, :])
        target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)
        y_pred = model(input_values.transpose(0, 1).view(1, q, dim))
        state_pred_train[:, i - TrainLen] = y_pred.data.numpy()
        # Rest of iters - change iter by sliding back, overwriting last with newly predicted state_predv2
        for i in range(TrainLen + 1, TotLen):
            input_values[:, :-1] = input_values[:, 1:]
            input_values[:, -1] = torch.tensor(state_pred_train[:, i - TrainLen - 1])
            target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

            y_pred = model(input_values.transpose(0, 1).view(1, q, dim))
            state_pred_train[:, i - TrainLen] = y_pred.data.numpy()
        auto_store[e, :, :]=np.transpose(state_pred_train)


# Given trained net, get training and testing performance
# Training and Teacher Forcing performance
trainloss = np.zeros(TrainLen+TestLen)
state_pred = np.zeros((dim,TrainLen+TestLen))
last_hidden = np.zeros((h_size, TrainLen + TestLen))
for i in range(TrainLen + TestLen):
    input_values = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    # Get and store final hidden state
    out, (h, c) = model[0](input_values.transpose(0, 1).view(1, q, dim))
    last_hidden[:, i] = out[0, -1, :].data.numpy()  # Final hidden state

    y_pred = model(input_values.transpose(0, 1).view(1, q, dim))
    loss = loss_fn(y_pred, target)
    trainloss[i] = loss.item()
    state_pred[:, i] = y_pred.data.numpy()

# Autonomous Mode
test_loss = np.zeros(TestLen)
state_pred_v2 = np.zeros((dim, TestLen))
i = TrainLen
input_values = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
input_values[:, -1] = torch.tensor(state_pred[:, TrainLen - 1])
target = torch.tensor(np.copy(states[:, i+q]), dtype=torch.float)
y_pred = model(input_values.transpose(0, 1).view(1, q, dim))
loss = loss_fn(y_pred, target)
test_loss[i - TrainLen] = loss.item()
state_pred_v2[:, i - TrainLen] = y_pred.data.numpy()
# Rest of iters - change iter by sliding back, overwriting last with newly predicted state_predv2
for i in range(TrainLen + 1, TotLen):
    input_values[:, :-1] = input_values[:, 1:]
    input_values[:, -1] = torch.tensor(state_pred_v2[:, i - TrainLen - 1])
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    # Get and store final hidden state
    out, (h, c) = model[0](input_values.transpose(0, 1).view(1, q, dim))
    last_hidden[:, i] = out[0, -1, :].data.numpy()  # Final hidden state

    y_pred = model(input_values.transpose(0, 1).view(1, q, dim))
    loss = loss_fn(y_pred, target)
    test_loss[i - TrainLen] = loss.item()
    state_pred_v2[:, i - TrainLen] = y_pred.data.numpy()



# So instead, try the above code found from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
# Given PREVIOUS hidden state (e.g. h_(q-1)), extract from the network the transform h_(q-1) to h_q - ?? This is the LSTM hidden layer update term, neglecting inputs?
# Then, given the function (LSMT), previous state (x) and number of outputs(hsize), above should give jacobian! But only for h_(q-1) to h_q...
# as a test, try it with current 'out', out [0,-1,:] is last state, [0,-2,:] is previous e.g. our x
# But how to get this? model[0].weight_hh_l0 gives us a 200x50 vector...
# Could instead just use the LSTM with a given hidden state, but this neglects the cell? Try it anyways...
# model[0](input_values[:, -1].view(1, 1, 3), (out[0, -2, :].view(1, 1, 50), (out[0, -2, :] * 0).view(1, 1, 50)))[0] # does not match...
# Presumably using the wrong c0 messes it up. We could potentially extract the 'true' c0 to use
out0, (h0, c0) = model[0](input_values[:, :-1].transpose(0, 1).view(1, q - 1, dim))
# model[0](input_values[:, -1].view(1, 1, 3), (h0, c0))[0] # now matches
# The fact that we require additional inputs means we can't use the above code as is
# Instead, try to redo by hand?
n = 50
testh0 = h0.squeeze()
testh0 = testh0.repeat(50,1)
testh0.requires_grad_(True)
test_in = input_values[:, -1].view(1, 1, 3)
test_in.requires_grad = True
testy = model[0](test_in, (h0, c0))[0][0, 0, 0]
optimizer.zero_grad()
testy.backward(torch.eye(50), retain_graph=True)
# h0.grad.data # fails...
# Seems to only compute gradient for the inputs, not with respect to hidden state, so useless to us
# That said, can probably explicitly compute (no autograd unforunately) the ESN one, since hidden states ARE in the comp graph
# What about instead, doing eigenvalues direclty on output e.g. d(xnext)/dx, rather than on hidden? LSTM autograd may work here
# Model output is 1x1x50 - multiply the 50 by the 50x3 final output to get total output?
J_test = np.zeros((3, 3))
test_x_1 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[0] # first output
# optimizer.zero_grad()
test_x_1.backward(torch.eye(3), retain_graph=True)
# testin.grad # exists! This should now be the first row of the jacobian - repeat for rows 2 and 3
J_test[0, :] = test_in.grad.data.numpy()[0, 0, :]
test_x_2 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[1] # 2nd output
optimizer.zero_grad()
test_x_2.backward(torch.eye(3), retain_graph=True)
J_test[1, :] = test_in.grad.data.numpy()[0, 0, :]
test_x_3 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[2] # 3rd output
optimizer.zero_grad()
test_x_3.backward(torch.eye(3), retain_graph=True)
J_test[2, :] = test_in.grad.data.numpy()[0, 0, :]
# Is this enough? This is the eigenvalues of the ONE step ahead prediction, given the hidden state for q>1
# Should this be done on TF or auto?


Jac_TF = np.zeros((TestLen, 3, 3))
Jac_eig_values = np.zeros((TestLen, 3, 2))  # real/imag
Jac_eig_vectors = np.zeros((TestLen, 3, 3))
Lor_eig_values = np.zeros((TestLen, 3, 2))
Lor_eig_vectors = np.zeros((TestLen, 3, 3))
for i in range(TrainLen, TrainLen + TestLen):
    input_values = torch.tensor(np.copy(states[:, i:i + q]), dtype=torch.float)
    target = torch.tensor(np.copy(states[:, i + q]), dtype=torch.float)

    out0, (h0, c0) = model[0](input_values[:, :-1].transpose(0, 1).view(1, q - 1, dim))
    test_in = input_values[:, -1].view(1, 1, 3)
    test_in.requires_grad = True
    J_test = np.zeros((3, 3))
    test_x_1 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[0]  # first output
    optimizer.zero_grad()
    test_x_1.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    J_test[0, :] = test_in.grad.data.numpy()[0, 0, :]
    test_x_2 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[1]  # 2nd output
    optimizer.zero_grad()
    test_x_2.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    J_test[1, :] = test_in.grad.data.numpy()[0, 0, :]
    test_x_3 = torch.matmul(model[0](test_in, (h0, c0))[0][-1, -1, :], model[2].weight.t())[2]  # 3rd output
    optimizer.zero_grad()
    test_x_3.backward(torch.eye(3, dtype=torch.float), retain_graph=True)
    J_test[2, :] = test_in.grad.data.numpy()[0, 0, :]

    # Store Jacobian
    Jac_TF[i - TrainLen, :, :] = J_test - np.eye(3)
    # Get and store eig_values, eig_vectors
    J_eig_values, J_eig_vectors = np.linalg.eig(J_test - np.eye(3))
    Jac_eig_values[i - TrainLen, :, 0] = np.real(J_eig_values)
    Jac_eig_values[i - TrainLen, :, 1] = np.imag(J_eig_values)
    Jac_eig_vectors[i - TrainLen, :, :] = J_eig_vectors

    # Get and store true (Lorenz) Jacobean, eigen_val, eigen_vectors
    Jl = np.zeros((3, 3))  # Eigenvalues of lorenz system
    sl = states[:, -2000 + i] * regularized
    Jl[0, 0] = -sigma
    Jl[0, 1] = sigma
    Jl[0, 2] = 0
    Jl[1, 0] = rho - sl[2]
    Jl[1, 1] = -1
    Jl[1, 2] = -sl[0]
    Jl[2, 0] = sl[1]
    Jl[2, 1] = sl[0]
    Jl[2, 2] = -beta
    Jl[0, :] /= regularized[0]
    Jl[1, :] /= regularized[1]
    Jl[2, :] /= regularized[2]
    Jl *= .25
    l_eig_values, l_eig_vectors = np.linalg.eig(Jl)
    Lor_eig_values[i - TrainLen, :, 0] = np.real(l_eig_values)
    Lor_eig_values[i - TrainLen, :, 1] = np.imag(l_eig_values)
    Lor_eig_vectors[i - TrainLen, :, :] = l_eig_vectors

num_plot = TestLen
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('Lorenz TF Eigenvalues, Test step = 0')
ax = fig.add_subplot(111)  # 221 if we want 4
plot = ax.scatter(Jac_eig_values[0, :, 0], Jac_eig_values[0, :, 1])


def update4(frame, plot_in, Jaceigvals):
    fig.suptitle('Lorenz TF Eigenvalues, Test step = ' + str(frame))
    ax.clear()
    plot_in = ax.scatter(Jaceigvals[frame, :, 0], Jaceigvals[frame, :, 1], c='b')
    ax.scatter(Lor_eig_values[frame, :, 0], Lor_eig_values[frame, :, 1], c='r')
    ax.set_ylim(-.5, .5)
    ax.set_xlim(-1, .5)
    ax.grid(True)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend(["LSTM Eigvals", "Lorenz Eigvals"])
    return plot_in,


ani = animation.FuncAnimation(fig, update4, num_plot, fargs=(plot, Jac_eig_values))
ani.save('LSTM_TFEig.mp4', writer='ffmpeg', fps=20)
