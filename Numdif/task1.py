# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:04:29 2022

"""
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
import time

#%%
def plot_solution(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation. This function was given in exercise 2
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    T, X = np.meshgrid(t,x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt)
#%%
def European(x, K, H):
    return max(K - x, 0)
def Butterfly(x, K, H):
    return max(0, x - K) - 2*max(0, x-K-H) + max(0,x-K-2*H)
def Binary(x, K, H):
    return max(0, np.sign(x - K))

Euro = np.vectorize(European)
Butter = np.vectorize(Butterfly)
Bin = np.vectorize(Binary)


#%%
#Forward Euler, Boundary condition 1
@jit
def FEBC1(R, T, M, N, sigSq, r, c, K, H, f0):
    '''
    solving BS with FE and BC1
    ------
    R: x-boundary
    T: end-time
    M: # of internal nodes
    N: # of time steps
    sigSq: volatility
    r: interest rate
    c: dividends
    f0: function for calculating initial condition
    '''
    h = R/(M+1) #step length in space
    k = T/N #step length in time
    x = np.arange(0, R+0.5*h, h) #space-points, including end points
    t = np.arange(0, T+0.5*k, k) #time points, including end points
    
    U0 = f0(x, K, H) #initial condition
    
    n = np.arange(0,N+1,1)#vector with time indexes
    m = np.arange(0, M + 2, 1) #vector of where m[j] = (j+1)
    mSq = m**2  #vector of where mSq[j] = (j+1)^2
    
    lower = 0.5*( k*sigSq*mSq - k*r*m) #lower diagonal in A
    diag = -k*sigSq*mSq + k*c #diagonal in A
    upper = 0.5*( k*sigSq*mSq + k*r*m)#upper diagonal in A
    
    P = diags([lower[2:-1], diag[1:-1], upper[1:-2]], [-1,0,1])
    A = np.identity(M) + P #iteration matrix
    
    B = np.zeros((M, N + 1)) #matrix with modified BC's to use as b^n
    
    B[0] = U0[0]*np.exp(-c*k*n)*0.5*(k*sigSq - k*r)
    B[-1] = U0[-1]*np.ones(N+1)*0.5*(k*sigSq*M**2 + k*r*M)
    
    U = np.zeros((M+2, N+1)) 
    U[:,0] = U0
    U[0] = U0[0]*np.exp(-c*k*n) #setting BC
    U[-1] = U0[-1]*np.ones(N+1)

    for i in range(1, N+1):
        U[1:-1,i] = A.dot(U[1:-1, i-1]) + B[:,i-1]
    
    return x, t, U
    
#%%
#Backwards Euler, boundary condition one
@jit
def BEBC1(R, T, M, N, sigSq, r, c, K, H, f0):
    '''
    solving BS with BE and BC1
    ------
    R: x-boundary
    T: end-time
    M: # of internal nodes
    N: # of time steps
    sigSq: volatility
    r: interest rate
    c: dividends
    f0: function for calculating initial condition
    '''
    h = R/(M+1) #step length in space
    k = T/N #step length in time
    x = np.arange(0, R+0.5*h, h) #space-points, including end points
    t = np.arange(0, T+0.5*k, k) #time points, including end points
    
    U0 = f0(x, K, H) #initial condition
    
    n = np.arange(0,N+1,1)#vector with time indexes
    m = np.arange(0, M + 2, 1) #vector of where m[j] = (j+1)
    mSq = m**2  #vector of where mSq[j] = (j+1)^2
    
    lower = 0.5*( -k*sigSq*mSq + k*r*m) #lower diagonal in A
    diag = k*sigSq*mSq + k*c #diagonal in A
    upper = 0.5*( -k*sigSq*mSq - k*r*m)#upper diagonal in A
    
    P = diags([lower[2:-1], diag[1:-1], upper[1:-2]], [-1,0,1])
    A = np.identity(M) + P #iteration matrix
    
    B = np.zeros((M, N + 1)) #matrix with modified BC's to use as b^n
    
    B[0] = U0[0]*np.exp(-c*k*n)*0.5*(k*sigSq - k*r)
    B[-1] = U0[-1]*np.ones(N+1)*0.5*(k*sigSq*M**2 + k*r*M)
    
    U = np.zeros((M+2, N+1)) 
    U[:,0] = U0
    U[0] = U0[0]*np.exp(-c*k*n)#setting BC
    U[-1] = U0[-1]*np.ones(N+1)

    for i in range(1, N+1):
        U[1:-1,i] = np.linalg.solve(A, U[1:-1, i-1] + B[:,i] )
    
    return x, t, U

#%%
@jit
def AllBC1(R, T, M, N, sigSq, r, c, K, H, f0, method=1):
    '''
    solving BS with BE and BC1
    ------
    R: x-boundary
    T: end-time
    M: # of internal nodes
    N: # of time steps
    sigSq: volatility
    r: interest rate
    c: dividends
    f0: function for calculating initial condition
    method: which method to use. 0:FE, 1: BE, 2:CN
    '''
    thetas = [[0,0.5], [0.5, 0], [0.25, 0.25]]
    theta = thetas[method]
    
    h = R/(M+1) #step length in space
    k = T/N #step length in time
    x = np.arange(0, R+0.5*h, h) #space-points, including end points
    t = np.arange(0, T+0.5*k, k) #time points, including end points
    
    U0 = f0(x, K, H) #initial condition
    
    n = np.arange(0,N+1,1)#vector with time indexes
    m = np.arange(0, M + 2, 1) #vector of where m[j] = (j+1)
    mSq = m**2  #vector of where mSq[j] = (j+1)^2
    
    lower = ( k*sigSq*mSq - k*r*m) #lower diagonal in A
    diag = -2*k*sigSq*mSq -2*k*c #diagonal in A
    upper = ( k*sigSq*mSq + k*r*m)#upper diagonal in A
    
    P = diags([lower[2:-1], diag[1:-1], upper[1:-2]], [-1,0,1])
    A1 = np.identity(M) - theta[0]*P #iteration matrices
    A2 = np.identity(M) + theta[1]*P
    
    B = np.zeros((M, N + 1)) #matrix with modified BC's to use as b^n+1
    
    B[0] = U0[0]*( theta[0]*np.exp(-c*k*n) + theta[1]*np.exp(-c*k*(n-1) ))*(lower[1])
    B[-1] = U0[-1]*np.ones(N+1)*0.5*(upper[M])
    
    U = np.zeros((M+2, N+1)) 
    U[:,0] = U0
    U[0] = U0[0]*np.exp(-c*k*n)#setting BC
    U[-1] = U0[-1]*np.ones(N+1)

    for i in range(1, N+1):
        U[1:-1,i] = np.linalg.solve(A1, A2.dot( U[1:-1, i-1] + B[:,i]).A1 ) #.A1 is to convert from matric to vector
    
    return x, t, U


#%%
#Test of FEBC1/BEBC1
R = 50
T = 2
M = 1000
N = 10
sigSq = 0.1
r = 0.03
c = 0.05
K = 12
H = 8
#%%
#Test of combined function
xF, tF, UF = FEBC1(R, T, M, N, sigSq, r, c, K, H, Butter)
xFC, tFC, UFC = AllBC1(R, T, M, N, sigSq, r, c, K, H, Butter, method=2)

plot_solution(xF, tF, UF)
plot_solution(xFC, tFC, UFC)
#%%
tF1 = time.time()
xF, tF, UF = AllBC1(R, T, M, N, sigSq, r, c, K, H, Butter, method=0)
dtF = time.time()- tF1

tB1 = time.time()
xB, tB, UB = AllBC1(R, T, M, N, sigSq, r, c, K, H, Butter, method=1)
dtB = time.time() - tB1

tCN1 = time.time()
xCN, tCN, UCN = AllBC1(R, T, M, N, sigSq, r, c, K, H, Butter, method=2)
dtCN = time.time() - tCN1

print(f"Elapsed time: FE: {dtF}, BE:{dtB}, CN:{dtCN}")
paramtext = f"R:{R}, T:{T}, M:{M}, N:{N}, $\sigma ^2$:{sigSq}, r:{r}, c:{c}, K:{K}, H:{H}"
plot_solution(xF, tF, UF, txt="FE")
plot_solution(xB, tB, UB, txt="BE")
plot_solution(xCN, tCN, UCN, txt="CN")
#%%
#Test of initial condition functions
x = np.arange(0,70,7)

K = 10
H = 8
plt.plot(x, Euro(x, K, H))
plt.plot(x, Butter(x, K, H))
plt.plot(x, Bin(x, K, H))
plt.legend(["Euro", "Butterfly", "Binary"])

#%%
test = diags([[-2, -3], [3, 2, 1], [1, 2]], [-1,0,1])
test2 = np.identity(3) + test
print(test2)