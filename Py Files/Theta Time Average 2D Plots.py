''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
xi = 1.00           # Xi value
m = 0.00            # m value
r0 = 0.75           # Initial condition for rho not
o0 = 0.00           # Initial condition for theta
ti = 0              # Initial time value
tf = 120            # Final time value
qi = -2             # Minimum zeeman shift value
qf = 2              # Maximum zeeman shift value
ic = [r0, o0]       # Initial condition array
time = [ti, tf]     # Time array
zman = [qi, qf]     # Zeeman shift array

''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''
zeemanPartition = 100     # Zeeman shift partition size
zeemanSpace = np.linspace(zman[0], zman[-1], zeemanPartition)     # Zeemanshift space
timeSpace = np.linspace(time[0], time[-1], zeemanPartition)       # Time Space

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
sol = np.zeros((zeemanPartition, zeemanPartition))      # Solution Matrix
solT = np.zeros(zeemanPartition)

for i in range(zeemanPartition):
    sol[:, i] = odeint(SolveMe, ic, timeSpace, args=(zeemanSpace[i],))[:, 1]
    solT[i] = (1/time[-1])*np.sum(np.cos(sol[:, i]))*(timeSpace[-1]/zeemanPartition)
        
''' ----- ----- ----- ----- Plots ----- ----- ----- ----- '''

plt.plot(zeemanSpace, solT, label='$\u03C1_{0}=$ ' + str(ic[0]))
plt.title('Time Average $\u0398$ w.r.t Zeemanshift')
plt.ylabel('Time Average $\u0398$')
plt.xlabel('Zeemanshift q')
plt.legend(loc='upper right')
plt.show()

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''