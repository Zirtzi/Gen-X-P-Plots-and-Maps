''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
rhoPartition = 25         # Rho space partition size
thetaPartition = 25       # Theta space partition size
timePartition = 100       # Time space partition size
xi = 1.00                 # Xi value
m = 0.00                  # m value
q = 1.0                   # Zeemanshift value
ri = 0.00                 # Initial value for initical condition for rho not
rf = 0.75                 # Final value for initial condition for rho not
ui = 0                    # Initial condition for theta
uf = (np.pi)/2            # Initial condition for theta   
ti = 0                    # Initial time value
tf = 30                   # Final time value
rho = [ri, rf]            # Rho array
theta = [ui, uf]          # Theta array
time = [ti, tf]           # Time array

''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
rhoSpace = np.linspace(rho[0], rho[-1], rhoPartition)           # Rho space
thetaSpace = np.linspace(theta[0], theta[-1], thetaPartition)   # Theta space
timeSpace = np.linspace(time[0], time[-1], timePartition)       # Time Space

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''

''' ----- ----- ----- ----- Diffy Q's ----- ----- ----- ----- '''
def SolveMe(IC, t):
    r, u = IC
    pieceOne = (1-r)
    pieceTwo = (1-2*r)
    pieceThree = np.sqrt(pieceOne**2-m**2)
    dr = 2*xi*np.sin(u)*r*pieceThree
    du = 2*xi*(pieceTwo + (np.cos(u)*(pieceOne*pieceTwo-m**2))/(pieceThree)) - q
    f = [dr, du]
    return f

''' ----- ----- ----- ----- Solution Matrix and Whatnot ----- ----- ----- ----- '''
timeAvgSol = np.zeros((rhoPartition, thetaPartition))

for i in range(thetaPartition):
    for j in range(rhoPartition):
        flurdBop = odeint(SolveMe, [rhoSpace[j],thetaSpace[i]], timeSpace)[:,0]
        timeAvgSol[j,i] = ((1/time[-1])*np.sum(flurdBop)*(timeSpace[-1]/timePartition))
        
plt.figure()
plt.contourf(thetaSpace, rhoSpace, timeAvgSol, levels=300)
plt.title(r"$\bar{\rho}$ w.r.t $\rho_{0}$ & $\Theta_{0}$ w/ q = " + str(q))
plt.ylabel(r"$\rho_{0}$")
plt.xlabel(r"$\Theta_{0}$")
plt.colorbar()
plt.show()     


