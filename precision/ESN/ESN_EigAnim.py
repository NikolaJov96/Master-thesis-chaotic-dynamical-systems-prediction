# Animation of ESN eigenvalues in autonomous mode after being trained
# Based on default ESN params as of 4/7 - e.g. I unit activation, squaring every other unit

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


np.random.seed(1802)

actfn = lambda x: np.tanh(x)
actfn = lambda x: x # Works just as well

def nonlin(x):
    x2 = np.copy(x)
    if len(np.shape(x2))==2:
        for i in range(np.shape(x)[1] // 2):
            x2[:, 2 * i] = (x2[:, 2 * i] ** 2).copy()
        return x2
    else: # assuming len = 1
        for i in range(len(x2) // 2):
            x2[2 * i] = (x2[2 * i] ** 2).copy()
    return x2


N = 100
Ntrain = 50000
Ntest = 2000

rho_l = 28.0
sigma_l = 10.0
beta_l = 8.0 / 3.0
dim = 3
def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma_l * (y - x), x * (rho_l - z) - y, x * y - beta_l * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, (Ntrain+Ntest)/200 + .005, 0.005)

states = odeint(f, state0, t)

# Center and Rescale
#states -= np.mean(states,0)
#states /= np.max([[np.max(states,0)],[-np.min(states,0)]],0) # rescale to -1,1
#states /= np.std(states) # Rescale to std = 1
regularized = np.std(states,0)
states /= np.std(states,0)

useHenon = False
if useHenon:
    dim = 2  # Henon
    # Generate data - Using Henon Map
    alpha = 1.4
    b = .3
    # Xn+1 = 1 - alpha * xn**2 + yn
    # Yn+1 = b*xn
    states = np.zeros((dim, Ntrain + Ntest + 1))
    for i in range(1, Ntrain + Ntest + 1):
        states[:, i] = np.array(
            [1 - alpha * states[0, i - 1] * states[0, i - 1] + states[1, i - 1], b * states[0, i - 1]])

    states = np.transpose(states)/np.std(np.transpose(states), 0)


p = np.min([3/N,1.0]) # they use 3/N
rho = .1 # default of .1 - rho as high as 1.25 works pretty well - mediocre results at intermidiate rho
# from .05 to .25 is similar to default. Middling ranges can give worse results. Results start to IMPROVE as rho above one, e.g. 1.15 is nearly 2x as good, continues to imrpve up to 1.75
A = np.random.randn(N,N)
# They use .rand? despite .randn giving better results... at their default rho. At extreme rho e.g. 2+, rand works better
A = np.random.rand(N,N)
Amask = np.random.rand(N,N)
A[Amask>p] = 0
[eigvals,eigvecs] = np.linalg.eig(A)
A/=(np.real(np.max(eigvals)))
A*=rho
A*=1
A*=0

Win = np.random.rand(dim,N)*2 - 1 # dense
# Experimental - in their version, they make Win black structure, e.g. each neuron only receives one input, not mixed
# Turns out to be very beneficial
#Winc = np.copy(Win)
#for i in range(dim):
#    Win[i,i*(N//dim):(i+1)*(N//dim)]*=2
#Win-=Winc

sigma = .5 # from paper - scale of input. They use .5, although higher values appear to help (e.g. 1.5 > .5 performance) Very low (e.g. .05) hurt performance
Win = Win * sigma
# Generate network values - unroll output in time
rout = np.zeros((Ntrain,N))
# They initialize r0 = 0
r = np.zeros(N)
for i in range(Ntrain):
    r = actfn(A@r + states[i,:]@Win)
    rout[i,:] = r
    #  Add slight noise to either r or rout. Start with directly to r
    #r += (np.random.rand(N)*2 - 1)*(1e-5) # e-5 to e-6 works ok, but no large benefit
#trout = np.copy(rout)
#for i in range(N//2):
#    trout[:,2*i] = (trout[:,2*i]**2).copy()
trout = nonlin(rout)
#trout += (np.random.rand(Ntrain,N)*2-1)*(1e-5) # works decently here too

# Train offline  - min error norm + l2 error of trout*Wout - states, where Wout = Nx3
Id_n = np.identity(N)
beta = .0001
U = np.dot(trout.transpose(),trout) + Id_n * beta
Uinv = np.linalg.inv(U)
Wout = np.dot(Uinv,np.dot(trout.transpose(),states[1:Ntrain+1,:]))

# Try to use a lower dim version of Wout?? Not sure how to test this
simplify = False
if (simplify):
    test = 1

rstore = np.copy(r)
# Predictions - unroll reservoir for Ntest
rpred = np.zeros((Ntest,N))
rpred2 = np.zeros((Ntest,N))
r2 = np.copy(rstore)
r = np.copy(rstore)
for i in range(Ntest):
    r = actfn(A @ r + states[Ntrain+i, :] @ Win) + 0e-3*(np.random.rand(N)*2-1)# Teacher Forcing
    r3 = nonlin(r2)
    #r2 = actfn(A @ r2 + r3 @ Wout @ Win) + 1e-3*(np.random.rand(N)*2-1)
    nextin = r3 @ Wout + (np.random.rand(3)*2-1)*0e-3
    r2 = actfn(A @ r2 + nextin @ Win)
    rpred[i,:] = r
    rpred2[i,:] = r2
trpred = nonlin(rpred)
trpred2 = nonlin(rpred2)

ResPred = trpred @ Wout
errors = np.sqrt(np.sum((ResPred - states[Ntrain+1:,:])**2,1))
errorsRMSE = np.sqrt(np.mean((ResPred - states[Ntrain+1:,:])**2,1))

ResPred2 = trpred2 @ Wout
errors2 = np.sqrt(np.sum((ResPred2 - states[Ntrain+1:,:])**2,1))

ResTrain = trout@Wout
errors_og = np.sqrt(np.sum((trout@Wout - states[1:Ntrain+1,:])**2,1))

# Precompute normalized directions
# up/down in attractor 1: 32470, 32680
# across in attractor 1 : 32100, 32510
# up/down in attractor 2: 30000, 30080
# across in attractor 2 : 30052, 30102
A1D1 = np.array([states[32470,0]-states[32680,0],states[32470,1]-states[32680,1],states[32470,2]-states[32680,2]])
A1D2 = np.array([states[32100,0]-states[32510,0],states[32100,1]-states[32510,1],states[32100,2]-states[32510,2]])
A2D1 = np.array([states[30000,0]-states[30080,0],states[30000,1]-states[30080,1],states[30000,2]-states[30080,2]])
A2D2 = np.array([states[30052,0]-states[30102,0],states[30052,1]-states[30102,1],states[30052,2]-states[30102,2]])
A1D1 = A1D1/np.linalg.norm(A1D1)
A1D2 = A1D2/np.linalg.norm(A1D2)
A2D1 = A2D1/np.linalg.norm(A2D1)
A2D2 = A2D2/np.linalg.norm(A2D2)

# Randomize Wout to show it is critical
# Wout = (np.random.rand(100,3)*2-1)*.1

# Get Jacobians during testing, save, and animate
Jstore = np.zeros((Ntest,N))
Jlstore = np.zeros((Ntest,3))
eigstore = np.zeros((Ntest,N,2)) # Testing x Eigvals x Real/Imag
leigstore = np.zeros((Ntest,3,2))
eigvecstore = np.zeros((Ntest,N,4)) # Testing x Eigvecs x (A1D1...A2D2)
leigvecstore = np.zeros((Ntest,3,4)) # Same, but for Lorenz eigvecs
fleigvecs = np.zeros((Ntest,3,3)) # Story full set of eigenvecs
updatestore = np.zeros((Ntest,3,4)) # Ntest updates, 3 versions, 4 axes
for k in range(Ntest):
    r2 = rpred2[k,:]
    J = np.zeros((N,N))
    B = (Wout @ Win)
    for i in range(N):
        for j in range(N):
            if np.mod(j,2)==0:
                J[i,j] = A[i,j] + B[j,i]*2*r2[j] # Auto
                #J[i,j] = A[i,j] # TF
            else:
                J[i,j] = A[i,j] + B[j,i] # Auto
                #J[i, j] = A[i, j] # TF
    J-=np.eye(N)
    #np.linalg.eigvals(J - np.eye(100))
    # Try to get J for the 3x3 output? This is just Wout *
# Compute this for various r2s and test eigs? We have r2_i saved as rpred2[i,:]
    eigvals, eigvecs = np.linalg.eig(J)
    eigstore[k,:,0] = np.real(eigvals)
    eigstore[k,:,1] = np.imag(eigvals)
    Jl = np.zeros((3,3)) # Eigenvalues of lorenz system
    sl = states[-2000+k,:]*regularized
    Jl[0,0] = -sigma_l
    Jl[0,1] = sigma_l
    Jl[0,2] = 0
    Jl[1,0] = rho_l - sl[2]
    Jl[1,1] = -1
    Jl[1,2] = -sl[0]
    Jl[2,0] = sl[1]
    Jl[2,1] = sl[0]
    Jl[2,2] = -beta_l
    Jl[0,:] /= regularized[0]
    Jl[1,:] /= regularized[1]
    Jl[2,:] /= regularized[2]
    Jl*=.05
    #Jl+=np.eye(3)
    leigvals, leigvecs = np.linalg.eig(Jl)
    leigstore[k,:,0] = np.real(leigvals)
    leigstore[k,:,1] = np.imag(leigvals)

    # Get projections for top 3 eigenvecs in each attractor plane
    if k>1: # Test whether we need to reverse eigenvectors - they can randomly be + or -
        for i in range(N): # if column difference is smaller when reversed, then reverse
            if (np.sum(np.abs(eigvecs[:,i]-eigveclast[:,i])))>(np.sum(np.abs(-eigvecs[:,i]-eigveclast[:,i]))):
                eigvecs[:,i]*=-1
        for i in range(3): # same thing, but for lorenz
            if (np.sum(np.abs(leigvecs[:, i] - leigveclast[:, i]))) > (np.sum(np.abs(-leigvecs[:, i] - leigveclast[:, i]))):
                leigvecs[:, i] *= -1
    eigveclast = np.copy(eigvecs)
    leigveclast = np.copy(leigvecs)
    fleigvecs[k, :, :] = leigvecs
    projvec = (np.transpose(Wout) @ eigvecs) # 3 X N
    projvec = np.real(projvec)
    eigvecstore[k,:,0] = np.dot(np.transpose(projvec),A1D1)
    eigvecstore[k,:,1] = np.dot(np.transpose(projvec),A1D2)
    eigvecstore[k,:,2] = np.dot(np.transpose(projvec),A2D1)
    eigvecstore[k,:,3] = np.dot(np.transpose(projvec),A2D2)
    leigvecstore[k,:,0] = np.dot(np.transpose(leigvecs),A1D1)
    leigvecstore[k,:,1] = np.dot(np.transpose(leigvecs),A1D2)
    leigvecstore[k,:,2] = np.dot(np.transpose(leigvecs),A2D1)
    leigvecstore[k,:,3] = np.dot(np.transpose(leigvecs),A2D2)
    # Get projections of top3, all eigenvec*eigenvals to see if the other 97 are adding anything important
    esn3 = np.real(np.sum(projvec[:,0:3]*eigvals[0:3],1)) # top 3 esn
    esnall = np.real(np.sum(projvec * eigvals, 1)) # all esn
    lorenzall = np.real(np.sum(leigvecs*leigvals,1)) # all lorenz
    # Project them down into the attractor plains, and save
    updatestore[k, 0, 0] = np.dot(esn3, A1D1)
    updatestore[k, 0, 1] = np.dot(esn3, A1D2)
    updatestore[k, 0, 2] = np.dot(esn3, A2D1)
    updatestore[k, 0, 3] = np.dot(esn3, A2D2)
    updatestore[k, 1, 0] = np.dot(esnall, A1D1)
    updatestore[k, 1, 1] = np.dot(esnall, A1D2)
    updatestore[k, 1, 2] = np.dot(esnall, A2D1)
    updatestore[k, 1, 3] = np.dot(esnall, A2D2)
    updatestore[k, 2, 0] = np.dot(lorenzall, A1D1)
    updatestore[k, 2, 1] = np.dot(lorenzall, A1D2)
    updatestore[k, 2, 2] = np.dot(lorenzall, A2D1)
    updatestore[k, 2, 3] = np.dot(lorenzall, A2D2)


# Also need to get Jacobian of true Lorenz system at each state
# Lorenz given by
# x' = k(y-x), k = sigma_l
# y' = x(k-z) -y, k = rho_l
# z' = xy -kz, k = beta_l
# But we additionally divide each of x,y,z by a constant kx,ky,kz to convert from lorenz to ESN state space
# So final values of dx/d(all) divided by kx, etc
# J_ij = df_i/dx_j
# However, we want to compare to the Jacobian of the ESN, which we used from r(next) = f(rlast)
# For Lorenz, Rnext = Rlast + f(rlast)*dt, dt = .005
# So Jnew = I + .005*Jold

# Additionally, interestd in computing and showing eigenvectors
#eigvals,eigvecs = np.linalg.eig(J)
#eigvecs[:,0] @ Wout # From doc, eigvec[:,i] is ith eigenvector
#(np.transpose(Wout) @eigvecs)[:,0] # Gives equivelant answer, so this is how we compute them
# up/down in attractor 1: 32470, 32680
# across in attractor 1 : 32100, 32510
# up/down in attractor 2: 30000, 30080
# across in attractor 2 : 30052, 30102
# So, compute eigenvectors (which are normalized to unit length), then project them into our lorenz R^3 space, then dot product them with up/down or across vector in each attractor, and display?



numplot = Ntest
fig = plt.figure()
fig.set_size_inches(18, 10)
fig.suptitle('ESN Autonomous Eigenvalues, Test step = 0')
ax = fig.add_subplot(221) # 221 if we want 4
plot = ax.scatter(eigstore[0,:,0], eigstore[0,:,1])
ax.set_ylim(-.25, .25)
ax.set_xlim(-1, 2)
ax2 = fig.add_subplot(222, projection='3d')
plot2 = ax2.scatter(ResPred2[0, 0], ResPred2[0, 1], ResPred2[0, 2], 'r')
ax2.plot(states[Ntrain + 1:Ntrain + 2001, 0], states[Ntrain + 1:Ntrain + 2001:, 1], states[Ntrain + 1:Ntrain + 2001:, 2], 'b')
ax3 = fig.add_subplot(223)
plot3 = ax3.scatter(eigvecstore[0,:,1],eigvecstore[0,:,0])
ax4 = fig.add_subplot(224)
plot4 = ax4.scatter(eigvecstore[0,:,3],eigvecstore[0,:,2])

def update(frame,plotin, eigstore):
    fig.suptitle('ESN Autonomous Eigenvalues, Test step = '+str(frame))
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    plotin = ax.scatter(eigstore[frame,:,0], eigstore[frame,:,1])
    ax.scatter(leigstore[frame,:,0],leigstore[frame,:,1], c = 'r')
    ax.set_ylim(-.25, .25)
    ax.set_xlim(-1.5,.5)
    plotin2 = ax2.scatter(ResPred2[frame, 0], ResPred2[frame, 1], ResPred2[frame, 2], c = 'b')
    ax2.scatter(states[Ntrain+1+frame,0],states[Ntrain+1+frame,1],states[Ntrain+1+frame,2], c = 'r')
    ax2.plot(states[Ntrain + 1:Ntrain + 2001, 0], states[Ntrain + 1:Ntrain + 2001:, 1],states[Ntrain + 1:Ntrain + 2001:, 2], 'b')
    #plotin3 = ax3.scatter(eigvecstore[frame, 0:3, 1], eigvecstore[frame, 0:3, 0])
    #ax3.scatter(leigvecstore[frame, 0:3, 1], leigvecstore[frame, 0:3, 0], c = 'r')
    #plotin4 = ax4.scatter(eigvecstore[frame, 0:3, 3], eigvecstore[frame, 0:3, 2])
    #ax4.scatter(leigvecstore[frame, 0:3, 3], leigvecstore[frame, 0:3, 2], c = 'r')
    # Change from scattered points to arrows:
    plotin3 = ax3.plot(np.array([[0,0,0],eigvecstore[frame,0:3,1]]), np.array([[0,0,0],eigvecstore[frame,0:3,0]]))
    ax3.plot(np.array([[0,0,0],leigvecstore[frame,0:3,1]]), np.array([[0,0,0],leigvecstore[frame,0:3,0]]), 'r')
    plotin4 = ax4.plot(np.array([[0,0,0],eigvecstore[frame,0:3,3]]), np.array([[0,0,0],eigvecstore[frame,0:3,2]]))
    ax4.plot(np.array([[0,0,0],leigvecstore[frame,0:3,3]]), np.array([[0,0,0],leigvecstore[frame,0:3,2]]), 'r')
    # Additionally, scatter in True/large est/small est updates?
    ax3.scatter(updatestore[frame, 0, 1], updatestore[frame, 0, 0],c = 'g')
    ax3.scatter(updatestore[frame, 1, 1], updatestore[frame, 1, 0],c = 'b')
    ax3.scatter(updatestore[frame, 2, 1], updatestore[frame, 2, 0],c = 'r')
    ax4.scatter(updatestore[frame, 0, 3], updatestore[frame, 0, 2],c = 'g')
    ax4.scatter(updatestore[frame, 1, 3], updatestore[frame, 1, 2],c = 'b')
    ax4.scatter(updatestore[frame, 2, 3], updatestore[frame, 2, 2],c = 'r')
    #ax3.set_xlim(-.5,.5)
    #ax3.set_ylim(-.5,.5)
    #ax4.set_xlim(-.5,.5)
    #ax4.set_ylim(-.5,.5)
    ax3.set_xlim(-1,1) # Larger axis scale needed if including Lorenz as well
    ax3.set_ylim(-1,1)
    ax4.set_xlim(-1,1)
    ax4.set_ylim(-1,1)
    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax.set_title("Eigenvalues (Red: Lorenz, Blue: ESN)")
    ax2.set_title("Autonomous Prediction (Red: True, Blue : ESN)")
    ax3.set_title("Eigenvectors projected into 1st Attractor Plain")
    ax4.set_title("Eigenvectors projected into 1st Attractor Plain")
    return plotin,
ani = animation.FuncAnimation(fig, update, numplot, fargs = (plot, eigstore))
ani.save('./Pytorch/ESN/Figures/ESN_AutoEig_N5.mp4',writer='ffmpeg',fps=20)
# Changes from Auto to TF - changed from plotting respred2 to respred, ditto for getting state of ESN

# Test whether ESN eigs (both esn and lorenz) are swapping sign randomly - start with lorenz, is easier
# Visualize lorenz output across time - should see jumping
fig = plt.figure()
plt.plot(fleigvecs[:,0,0])
plt.plot(fleigvecs[:,1,0])
plt.savefig('./Pytorch/ESN/Figures/ESN_Eig_Fliptest.png') # no longer needed, new flip algo seems to fix (most?) problems

# Testing - output ESN predictions in TF
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(ResPred2[:,0],'r')
ax.plot(states[Ntrain+1:,0])
ax2 = fig.add_subplot(312)
ax2.plot(ResPred2[:,1],'r')
ax2.plot(states[Ntrain+1:,1])
ax3 = fig.add_subplot(313)
ax3.plot(ResPred2[:,2],'r')
ax3.plot(states[Ntrain+1:,2])
plt.savefig('./Pytorch/ESN/Figures/ESN_AutoPanels_N5.png')

# Same for 3D view
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(ResPred2[:,0],ResPred2[:,1], ResPred2[:,2],'r')
ax.plot(states[Ntrain+1:,0],states[Ntrain+1:,1],states[Ntrain+1:,2],'b')
plt.savefig('./Pytorch/ESN/Figures/ESN_Auto_N5.png')

# Calculate lyapunov exponent? Not sure how principled this is over the entire trajectory (expect 0 as it is 'long term' stable in its attractor region)
# Calculate by avg of log of largest eigenvalue (real part) for all t
# Calculate for both ESN, Lorenz, need to add 1 to all eigs first
# eigstore - 2000x100x2, 3rd dimension 1st is real, 2nd imag
# leigstore - 2000x3x2
np.mean(np.log(np.max(eigstore[:1000,:,0]+1,1))) #.00558, first 1k is .00397
np.mean(np.log(np.max(leigstore[:1000,:,0]+1,1))) # .00699, first 1k is .00558

# End of useful code
exit(0)

# Above is not quite right. We want to compute eigenvalues of the ESN x -> x+, not hidden to hidden+1
# Overall computation should be very similar?
# X+ = Wout * (A*r + Win * X), but now stuck with a 'dead' first term?
# Just assume r is a generic constant like we did with the LSTM?
# Then it should just be Win@Wout?
C = Win@Wout
# Then it is just a constant?
D = Wout @ Win
Deigvals, Deigvecs = np.linalg.eig(D)
# Same for A+D?
Aeigvals, Aeigvecs = np.linalg.eig(A)
ADeigvals, ADeigvecs = np.linalg.eig(A + D)
np.mean(Wout[0:100:2,0]) # Squared terms; 3 orders of magnitude less
np.mean(Wout[1:101:2,0]) # than the linear terms
# One idea would be to replace A+D with its low order (or even low dim projection), but how does that interact with nonlinearity?
# Can't test low dim projection, can test A+D replaced with low order version
ADsm = (np.real(ADeigvals[0:3])*ADeigvecs[:,0:3])@np.linalg.pinv(ADeigvecs[:,0:3]) #???
ADsm = np.real(ADsm)

rpredAsm = np.zeros((Ntest,N))
rpredAsm2 = np.zeros((Ntest,N))
r2 = np.copy(rstore)
r = np.copy(rstore)
for i in range(Ntest):
    r = actfn(states[Ntrain+i, :] @ Win)
    r3 = nonlin(r2)
    nextin = r3 @ Wout
    r2 = actfn(nextin @ Win)
    rpredAsm[i,:] = r
    rpredAsm2[i,:] = r2
trpredAsm = nonlin(rpredAsm)
trpredAsm2 = nonlin(rpredAsm2)
AsmAutoTest = trpredAsm2 @ Wout
AsmTFTest = trpredAsm @ Wout

fig = plt.figure()
plt.plot(AsmTFTest[:,0],'r')
plt.plot(states[-2000:,0],'b')
# plt.savefig('./Pytorch/ESN/Figures/ESN_Auto_AsmTest.png') # Works fairly well, despite not being trained on this task in TF mode (auto is horrible)

# What if instead we make a tiny perturbation to A, Win and or Wout? How sensitive are we to these components
# In both TF and Auto mode
Wout_eps = np.copy(Wout) * (1+(np.random.rand(N,3)*2-1)*.00) + (np.random.rand(N,3)*2-1)*0
A_eps = np.copy(A) * (1+(np.random.rand(N,N)*2-1)*.00) + (np.random.rand(N,N)*2-1)*0
Win_eps = np.copy(Win) * (1+(np.random.rand(3,N)*2-1)*.00) + (np.random.rand(3,N)*2-1)*0
#A_eps = np.random.rand(N,N)
#Amask_eps = np.random.rand(N,N)
#A_eps[Amask_eps>p] = 0
#[eigvals_eps,eigvecs_eps] = np.linalg.eig(A_eps)
#A_eps/=(np.real(np.max(eigvals_eps)))
#A_eps*=rho
rpred_eps = np.zeros((Ntest,N))
rpred2_eps = np.zeros((Ntest,N))
r2 = np.copy(rstore)
r = np.copy(rstore)
for i in range(Ntest):
    r = actfn(A_eps @ r + states[Ntrain+i, :] @ Win_eps)
    r3 = nonlin(r2)
    nextin = r3 @ Wout_eps
    r2 = actfn(A @ r2 + nextin @ Win_eps)
    rpred_eps[i,:] = r
    rpred2_eps[i,:] = r2
trpred_eps = nonlin(rpred_eps)
trpred2_eps = nonlin(rpred2_eps)
auto_eps = trpred2_eps @ Wout_eps
tf_eps = trpred_eps @ Wout_eps

fig = plt.figure()
fig.set_size_inches(18, 10)
ax = fig.add_subplot(221, projection = '3d')
ax.plot(tf_eps[:,0],tf_eps[:,1], tf_eps[:,2],'r')
ax.plot(states[Ntrain+1:,0],states[Ntrain+1:,1],states[Ntrain+1:,2],'b')
ax2 = fig.add_subplot(222)
ax2.plot(tf_eps[:,0],'r')
ax2.plot(states[Ntrain+1:,0])
ax3 = fig.add_subplot(223)
ax3.plot(tf_eps[:,1],'r')
ax3.plot(states[Ntrain+1:,1])
ax4 = fig.add_subplot(224)
ax4.plot(tf_eps[:,2],'r')
ax4.plot(states[Ntrain+1:,2])
plt.title("ESN w/ eps perturbation, TF Mode")
plt.savefig('./Pytorch/ESN/Figures/ESN_EpsTF.png')

fig = plt.figure()
fig.set_size_inches(18, 10)
ax = fig.add_subplot(221, projection = '3d')
ax.plot(auto_eps[:,0],auto_eps[:,1], auto_eps[:,2],'r')
ax.plot(states[Ntrain+1:,0],states[Ntrain+1:,1],states[Ntrain+1:,2],'b')
ax2 = fig.add_subplot(222)
ax2.plot(auto_eps[:,0],'r')
ax2.plot(states[Ntrain+1:,0])
ax3 = fig.add_subplot(223)
ax3.plot(auto_eps[:,1],'r')
ax3.plot(states[Ntrain+1:,1])
ax4 = fig.add_subplot(224)
ax4.plot(auto_eps[:,2],'r')
ax4.plot(states[Ntrain+1:,2])
plt.title("ESN w/ eps perturbation, Autonomous Mode")
plt.savefig('./Pytorch/ESN/Figures/ESN_EpsAuto.png')