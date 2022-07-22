''' ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- '''
''' ----- ----- ----- ----- Module imports and what not ----- ----- ----- -----  '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

''' ----- ----- ----- ----- I.C and Global Variables: ----- ----- ----- ----- '''
zeemanPartition = 25  # Zeeman shift partition size
rhoPartition = 25  # Rho not space partition size
timePartition = 100  # Time space partition size
xi = 1.00  # Xi value
m = 0.00  # m value
ti = 0  # Initial time value
tf = 30  # Final time value
time = [ti, tf]  # Time array


''' ----- ----- ----- ----- Parameter Space and Partition Size ----- ----- ----- ----- '''

timeSpace = np.linspace(time[0], time[-1], timePartition)  # Time Space

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
timeAvgSol = np.zeros((rhoPartition, zeemanPartition))





