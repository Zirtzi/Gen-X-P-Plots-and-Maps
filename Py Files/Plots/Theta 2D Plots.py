''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
xi = 1.00  # Xi value
q = 2.00  # Zeemanshift value
m = 0.00  # m value
r0 = 0.75 # Initial condition for rho not
o0 = 0.00 # Initial condition for theta
ic = [r0, o0] # Initial condition array

''' ----- ----- ----- ----- Time interval of solution: ----- ----- ----- ----- '''
t = np.linspace(0, 6, 6000) # Time interval

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

''' ----- ----- ----- ----- Solution to ODE: ----- ----- ----- ----- '''
sol = odeint(SolveMe, ic, t) # Solution to ODE

''' ----- ----- ----- ----- Plots of solutions:  ----- ----- ----- -----'''
plt.plot(t, sol[:, 1], label = 'q = ' + str(q), color = 'darkorchid') # Theta plot
plt.title(r'$\Theta$ w.r.t Time w/ $\rho_{0}$ = ' + str(r0))
plt.xlabel(r'Time (s)')
plt.ylabel(r"$\Theta$")
plt.legend(loc='upper left')
plt.show() # Show plot
''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
