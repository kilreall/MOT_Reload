import numpy as np 
import matplotlib.pyplot as plt

def B(x, y, z):
    return np.sqrt(x**2+y**2+4*z**2)*A

def dwB(x, y, z, Ml, Mu):
    return (gu*Mu-gl*Ml)*B(x, y, z)*mB/h

def R(x, y, z, v, Ml, Mu, os):
    return Y/2*os/(1+4*(dw-2*np.pi*k*v/lam-dwB(x, y, z, Ml, Mu))/Y**2)

mRb = 1.46e-25
h = 1e-34
kb = 1.38e-23
Y = 2*np.pi*6.06e6
lam = 780e-9
k = 2*np.pi/lam
mB = 9.274e-24
gu = 2/3
gl = 1/2
g = 9.8

TK = 310 # Kelvin
dw = 13e6 # maybe * 2pi
A = 1600 # G/m - B grad
TMOL = 3e-3
T = 5e-3
TSIM = 2


dt = 1e-6
Nu = np.zeros(7)
Nl = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
vz = [np.random.normal(0, np.sqrt(kb*TK/mRb))]
z = [0]

os01 = 1/5

i = 0
while i <= int(TSIM/dt):
    for j in range(int(TMOL/dt)):
        az = h/mRb/lam*k( R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) )-g

