''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
zeemanPartition = 25  # Zeeman shift partition size
thetaPartition = 25  # Rho not space partition size
timePartition = 100  # Time space partition size
xi = 1.00  # Xi value
m = 0.00  # m value
ri = 0.50   # Initial value for rho
ti = 0  # Initial time value
tf = 30  # Final time value
qi = -4  # Initial value of zeeman shift
qf = 4   # Final value of zeeman shift
ui = 0   # Initial value for theta
uf = np.pi/4 # Finalve value for theta
time = [ti, tf]  # Time array
zeeman = [qi, qf]  # Zeeman array
theta = [ui, uf]   # Theta array


''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
timeSpace = np.linspace(time[0], time[-1], timePartition)  # Time Space
zeemanSpace = np.linspace(zeeman[0], zeeman[-1], zeemanPartition)  # Zeeman space
thetaSpace = np.linspace(theta[0], theta[-1], thetaPartition)   # Theta space

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''

''' ----- ----- ----- ----- Diffy Q's ----- ----- ----- ----- '''
def SolveMe(IC, t, args):
    r, u = IC
    pieceOne = (1 - r)
    pieceTwo = (1 - 2 * r)
    pieceThree = np.sqrt(pieceOne ** 2 - m ** 2)
    dr = 2 * xi * np.sin(u) * r * pieceThree
    du = 2 * xi * (pieceTwo + (np.cos(u) * (pieceOne * pieceTwo - m ** 2)) / (pieceThree)) - args
    f = [dr, du]
    return f

''' ----- ----- ----- ----- ----- Solutions ----- ----- ----- ----- ----- '''
timeAvgSol = np.zeros((thetaPartition, zeemanPartition))

for i in range(zeemanPartition):
    for j in range(thetaPartition):
        flurdBop = odeint(SolveMe, [ri,thetaSpace[j]], timeSpace, args=(zeemanSpace[i],))[:,1]
        timeAvgSol[j, i] = ((1/time[-1])*np.sum(np.cos(flurdBop/2))*(timeSpace[-1]/timePartition))

plt.figure()
plt.contourf(zeemanSpace, thetaSpace, timeAvgSol, levels=300)
plt.title(r"$\bar{\Theta}$ w.r.t $q$ and $\Theta_{0}$")
plt.xlabel(r"Zeeman shift $q$")
plt.ylabel(r"$\Theta_{0}$")
plt.colorbar()
plt.show()


