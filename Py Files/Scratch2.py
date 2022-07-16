''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
zeemanPartition = 20      # Zeeman shift partition size
timePartition = 100       # Time space partition size
rhoPartition = 20         # Rho not space partition size
thetaPartition = 20       # Theta space partition space
xi = 1.00                 # Xi value
m = 0.00                  # m value
ri = 0.00                 # Initial value for initical condition for rho not
rf = 0.75                 # Final value for initial condition for rho not
ui = 0.00                 # Initial condition for theta
uf = 1.00                 # Final value for initial condition for theta
ti = 0                    # Initial time value
tf = 30                   # Final time value
qi = -4                   # Minimum zeeman shift value
qf = 4                    # Maximum zeeman shift value
rho = [ri, rf]            # Rho not vector space
theta = [ui, uf]          # Theta vector space
time = [ti, tf]           # Time array
zman = [qi, qf]           # Zeeman shift array

''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
rhoSpace = np.linspace(rho[0], rho[-1], rhoPartition)           # Rho space
thetaSpace = np.linspace(theta[0], theta[-1], thetaPartition)   # Theta space
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
zeemanSol = np.zeros((timePartition, zeemanPartition))
thetaSol = np.zeros((rhoPartition, thetaPartition))
          
for i in range(thetaPartition):
    for j in range(rhoPartition):
        for k in range(zeemanPartition):
            zeemanSol[:, k] = odeint(SolveMe, [rhoSpace[j], thetaSpace[i]], timeSpace, args=(zeemanSpace[k],))[:, 0]
    thetaSol[:, i] = ((1/time[-1])*np.sum(np.cos(zeemanSol[:, i]/2))*(timeSpace[-1]/timePartition))
            
plt.figure()
plt.contourf(thetaSpace, rhoSpace, thetaSol, levels=300)
plt.title('Time Averaged $\u03C1$ for q: {' + str(zeemanSpace[0]) + ',' + str(zeemanSpace[-1]) + '}')
plt.ylabel('$\u03C1_{0}$')
plt.xlabel('$\u0398_{0}$')
plt.colorbar()
plt.show()
