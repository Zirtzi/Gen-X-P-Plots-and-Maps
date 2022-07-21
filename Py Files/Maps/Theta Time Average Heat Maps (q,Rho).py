''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
zeemanPartition = 50      # Zeeman shift partition size
rhoPartition = 50         # Rho not space partition size
timePartition = 100       # Time space partition size
xi = 1.00                 # Xi value
m = 0.00                  # m value
ri = 0.00                 # Initial value for initical condition for rho not
rf = 0.75                 # Final value for initial condition for rho not
ui = (np.pi)/4            # Initial condition for theta
ti = 0                    # Initial time value
tf = 30                   # Final time value
qi = -3                   # Minimum zeeman shift value
qf = 3                    # Maximum zeeman shift value
rho = [ri, rf]            # Rho not vector space
time = [ti, tf]           # Time array
zman = [qi, qf]           # Zeeman shift array

''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
rhoSpace = np.linspace(rho[0], rho[-1], rhoPartition)           # Rho space
zeemanSpace = np.linspace(zman[0], zman[-1], zeemanPartition)   # Zeemanshift space
timeSpace = np.linspace(time[0], time[-1], timePartition)       # Time Space

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''

''' ----- ----- ----- ----- Diffy Q's ----- ----- ----- ----- '''
def SolveMe(IC, t, args):
    r, u = IC
    pieceOne = (1-r)
    pieceTwo = (1-2*r)
    pieceThree = np.sqrt(pieceOne**2-m**2)
    dr = 2*xi*np.sin(u)*r*pieceThree
    du = 2*xi*(pieceTwo + (np.cos(u)*(pieceOne*pieceTwo-m**2))/(pieceThree)) - args
    f = [dr, du]
    return f

''' ----- ----- ----- ----- ----- Solutions ----- ----- ----- ----- ----- '''
zeemanSol = np.zeros((rhoPartition, zeemanPartition))
timeAvgSol = np.zeros((rhoPartition, zeemanPartition))
        
for i in range(zeemanPartition):
    for j in range(rhoPartition):
        zeemanSol[j, i] = odeint(SolveMe, [rhoSpace[j],ui], timeSpace, args=(zeemanSpace[i],))[len(timeSpace)-1,1]
        timeAvgSol[j, i] = ((1/time[-1])*np.sum(np.cos(zeemanSol[:, i]/2))*(timeSpace[-1]/timePartition))
        
plt.figure()
plt.contourf(zeemanSpace, rhoSpace, timeAvgSol, levels=300)
plt.title(r"Time Average $\Theta$ w.r.t Zeemanshift and $\rho_{0}$")
plt.xlabel("Zeeman shift q")
plt.ylabel("$\u03C1_{0}$")
plt.colorbar()
plt.show()
        



