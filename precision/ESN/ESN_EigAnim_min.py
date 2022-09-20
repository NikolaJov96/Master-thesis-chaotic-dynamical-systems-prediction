import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def act_fn(x):
    return x


def non_lin(x):
    x2 = np.copy(x)
    if len(np.shape(x2)) == 2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else:  # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
    return x2


def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives


np.random.seed(1802)

N = 100
N_train = 50000
N_test = 2000

rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, (N_train + N_test) / 200 + .005, 0.005)

states = odeint(f, state0, t)

# Center and Rescale
regularized = np.std(states, 0)
states /= np.std(states, 0)

p = np.min([3 / N, 1.0])  # they use 3/N
rho = .1
A = np.random.rand(N, N)
A_mask = np.random.rand(N, N)
A[A_mask > p] = 0
[eig_values, eig_vectors] = np.linalg.eig(A)
A /= (np.real(np.max(eig_values)))
A *= rho
A *= 1
A *= 0

Win = np.random.rand(dim, N) * 2 - 1  # dense

sigma = .5
Win = Win * sigma
rout = np.zeros((N_train, N))
r = np.zeros(N)
for i in range(N_train):
    r = act_fn(A @ r + states[i, :] @ Win)
    rout[i, :] = r

trout = non_lin(rout)

Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(), trout) + Id_n * beta
U_inv = np.linalg.inv(U)
Wout = np.dot(U_inv, np.dot(trout.transpose(), states[1:N_train + 1, :]))

r_store = np.copy(r)
r_pred = np.zeros((N_test, N))
r_pred_2 = np.zeros((N_test, N))
r2 = np.copy(r_store)
r = np.copy(r_store)
for i in range(N_test):
    r = act_fn(A @ r + states[N_train + i, :] @ Win) + 0e-3 * (np.random.rand(N) * 2 - 1)
    r3 = non_lin(r2)
    next_in = r3 @ Wout + (np.random.rand(3) * 2 - 1) * 0e-3
    r2 = act_fn(A @ r2 + next_in @ Win)
    r_pred[i, :] = r
    r_pred_2[i, :] = r2

# Get Jacobians during testing, save, and animate
eig_store = np.zeros((N_test, N, 2))  # Testing x Eig_values x Real/Imag
l_eig_store = np.zeros((N_test, 3, 2))
eig_vec_last = None
l_eig_vec_last = None
for k in range(N_test):
    r2 = r_pred_2[k, :]
    J = np.zeros((N, N))
    B = (Wout @ Win)
    for i in range(N):
        for j in range(N):
            if np.mod(j, 2) == 0:
                J[i, j] = A[i, j] + B[j, i] * 2 * r2[j]
            else:
                J[i, j] = A[i, j] + B[j, i]
    J -= np.eye(N)

    eig_values, eig_vectors = np.linalg.eig(J)
    eig_store[k, :, 0] = np.real(eig_values)
    eig_store[k, :, 1] = np.imag(eig_values)
    Jl = np.zeros((3, 3))
    sl = states[-2000 + k, :] * regularized
    Jl[0, 0] = -sigma_l
    Jl[0, 1] = sigma_l
    Jl[0, 2] = 0
    Jl[1, 0] = rho_l - sl[2]
    Jl[1, 1] = -1
    Jl[1, 2] = -sl[0]
    Jl[2, 0] = sl[1]
    Jl[2, 1] = sl[0]
    Jl[2, 2] = -beta_l
    Jl[0, :] /= regularized[0]
    Jl[1, :] /= regularized[1]
    Jl[2, :] /= regularized[2]
    Jl *= .05
    l_eig_values, l_eig_vectors = np.linalg.eig(Jl)
    l_eig_store[k, :, 0] = np.real(l_eig_values)
    l_eig_store[k, :, 1] = np.imag(l_eig_values)

    # Get projections for top 3 eigen_vectorss in each attractor plane
    if k > 1:  # Test whether we need to reverse eigenvectors - they can randomly be + or -
        for i in range(N):  # if column difference is smaller when reversed, then reverse
            if (np.sum(np.abs(eig_vectors[:, i] - eig_vec_last[:, i]))) > \
                    (np.sum(np.abs(-eig_vectors[:, i] - eig_vec_last[:, i]))):
                eig_vectors[:, i] *= -1
        for i in range(3):  # same thing, but for lorenz
            if (np.sum(np.abs(l_eig_vectors[:, i] - l_eig_vec_last[:, i]))) > \
                    (np.sum(np.abs(-l_eig_vectors[:, i] - l_eig_vec_last[:, i]))):
                l_eig_vectors[:, i] *= -1
    eig_vec_last = np.copy(eig_vectors)
    l_eig_vec_last = np.copy(l_eig_vectors)
    pro_vec = (np.transpose(Wout) @ eig_vectors)
    pro_vec = np.real(pro_vec)
    esn3 = np.real(np.sum(pro_vec[:, 0:3] * eig_values[0:3], 1))
    esn_all = np.real(np.sum(pro_vec * eig_values, 1))
    lorenz_all = np.real(np.sum(l_eig_vectors * l_eig_values, 1))


num_plot = N_test
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('ESN Autonomous Eigenvalues, Test step = 0')
ax = fig.add_subplot(111)
plot = ax.scatter(eig_store[0, :, 0], eig_store[0, :, 1])
ax.set_ylim(-.25, .25)
ax.set_xlim(-1, 2)


def update(frame, plot_in, eig_store_param):
    fig.suptitle('ESN Autonomous Eigenvalues, Test step = ' + str(frame))
    ax.clear()
    plot_in = ax.scatter(eig_store_param[frame, :, 0], eig_store_param[frame, :, 1])
    ax.scatter(l_eig_store[frame, :, 0], l_eig_store[frame, :, 1], c='r')
    ax.set_ylim(-.25, .25)
    ax.set_xlim(-1.5, .5)
    ax.grid(True)
    ax.set_title("Eigenvalues (Red: Lorenz, Blue: ESN)")
    return plot_in,


ani = animation.FuncAnimation(fig, update, num_plot, fargs=(plot, eig_store))
ani.save('ESN_AutoEig_N5.mp4', writer='ffmpeg', fps=20)
