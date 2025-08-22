import numpy as np
import matplotlib.pyplot as plt


mRb = 1.44e-25 # kg
kb = 1.23e-23
h = 1e-34
Y = 2*np.pi*6.06e6
k = 2*np.pi/780e-9
g = 9.68 # m/s2 gravuty acceleration

TK = 300 # Temperature K
s = 1
d = 3*Y # красная отстройка
mm = 0.93e6 # magnetic moment Hz
b = 1600 # G/m

dt = 1e-6 # simulation time step
N = int(1e6) # steps of sim

def dl(signk, v, z): # detuning 
    return d+signk*k*v+signk*mm*b*z

def fMOT(s, signk, v, z): # force
    return signk*h*k/2*s*Y/(1+s+(2*dl(signk, v, z)/Y)**2)

v = 0.1 #np.random.normal(0, np.sqrt(kb*TK/mRb))
z = 1e-3 #np.random.normal(0, 1e-3)
v = [v]
z = [z]
t = [0]

i = 0
while i <= N:
    for j in range(3000):
        v.append(v[-1]+dt*(fMOT(s, -1, v[-1], z[-1]) + fMOT(s, 1, v[-1], z[-1]) - mRb*g)/mRb)
        z.append(z[-1]+dt*v[-1])
        t.append((i+j)*dt)
    i += 3000

    for j in range(10000):
        v.append(v[-1]-dt*g)
        z.append(z[-1]+dt*v[-1])
        t.append((i+j)*dt)   
    i += 10000


v = np.array(v)
z = np.array(z)
t = np.array(t)

plt.plot(t*1e3, z*1e3)
plt.xlabel("time, [ms]")
plt.ylabel("position, [mm]")

plt.show()