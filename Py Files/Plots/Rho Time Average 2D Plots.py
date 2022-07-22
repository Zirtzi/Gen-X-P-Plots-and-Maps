''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
from numpy import gradient
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
xi = 1.00           # Xi value
m = 0.00            # m value
r0 = 0.75           # Initial condition for rho not
o0 = 0.00           # Initial condition for theta
ti = 0              # Initial time value
tf = 60             # Final time value
qi = -4             # Minimum zeeman shift value
qf = 4              # Maximum zeeman shift value
ic = [r0, o0]       # Initial condition array
time = [ti, tf]     # Time array
zman = [qi, qf]     # Zeeman shift array

''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
zeemanPartition = 100     # Zeeman shift partition size
timePartition = 10000
zeemanSpace = np.linspace(zman[0], zman[-1], zeemanPartition)     # Zeemanshift space
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

''' ----- ----- ----- ----- Solution Matrix ----- ----- ----- ----- '''
sol = np.zeros((timePartition, zeemanPartition))      # Solution Matrix
solT = np.zeros(zeemanPartition)                      # Time Average Matrix

for i in range(zeemanPartition):
    sol[:, i] = odeint(SolveMe, ic, timeSpace, args=(zeemanSpace[i],))[:, 0] 
    solT[i] = ((1/time[-1])*np.sum(np.cos(sol[:, i]/2))*(timeSpace[-1]/timePartition))

for j in range(1, len(solT)):
    if (np.absolute(solT[j]-solT[j-1])) >= 0.1:
        print("We have a jump here: " + str((zeemanSpace[j]+zeemanSpace[j-1])/2))
        
''' ----- ----- ----- ----- Derivatives of Data ----- ----- ----- ----- '''

dy = gradient(solT)
dx = gradient(zeemanSpace)
dydx = dy/dx

''' ----- ----- ----- ----- Plots ----- ----- ----- ----- '''
    
plt.plot(zeemanSpace, solT, label=r'$\rho_{0}=$ ' + str(ic[0]))
plt.plot(zeemanSpace, dydx, label=r"Derivative")
plt.title(r'$\bar{\rho}$ w.r.t Zeemanshift')
plt.ylabel(r'$\bar{\rho}$')
plt.xlabel('Zeemanshift q')
plt.legend(loc='upper right')
plt.show()

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''