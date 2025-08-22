import numpy as np

mRb = 1.46e-25
h = 6.62e-34
kb = 1.38e-23
Y = 2*np.pi*6.06e6
lam = 780e-9
k = 2*np.pi/lam

print(h*k/mRb/2/np.pi)