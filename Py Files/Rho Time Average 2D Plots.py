''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as It

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
xi = 1.00  # Xi value
q = 2.00  # Zeemanshift value
m = 0.00  # m value
r0 = 0.25 # Initial condition for rho not
o0 = 0.00 # Initial condition for theta
ic = [r0, o0] # Initial condition array
partition = 1000

''' ----- ----- ----- ----- Time interval of solution: ----- ----- ----- ----- '''
t = np.linspace(0, 6, partition) # Time interval

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''

''' ----- ----- ----- ----- Diffy Q's ----- ----- ----- -----  '''
def SolveMe(IC, t):
    r, u = IC
    pieceOne = (1-r)
    pieceTwo = (1-2*r)
    pieceThree = np.sqrt(pieceOne**2-m**2)
    dr = 2*xi*np.sin(u)*r*pieceThree
    du = 2*xi*(pieceTwo + (np.cos(u)*(pieceOne*pieceTwo-m**2))/(pieceThree)) - q
    f = [dr, du]
    return f

''' ----- ----- ----- ----- Time Average Theta: ----- ----- ----- ----- '''
solD = It.odeint(SolveMe, ic, t) # Solution to ODE
solT = (1/(t[-1]))*It.cumulative_trapezoid(np.cos(solD[:,0]), t) # Time average solution
interval = np.linspace(-2, 2, len(solT)) # X-axis interval

''' ----- ----- ----- ----- Plots ----- ----- ----- ----- '''
plt.plot(interval, solT, label='$\u03C1_{0}$ = ' + str(ic[0]))
plt.title('Time Average of $\u03C1$ w.r.t Zeeman Shift',)
plt.xlabel('Zeeman Shift (q)')
plt.ylabel('Time Average $\u03C1$')
plt.legend(loc='upper right')
plt.show()

''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
