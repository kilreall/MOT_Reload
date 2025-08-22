import numpy as np 
import matplotlib.pyplot as plt

def B(x, y, z): # маг поле
    return -2*z*A

def dwB(x, y, z, Ml, Mu):  # зеемановское смещение
    return (gu*Mu-gl*Ml)*B(x, y, z)*mB/(h/2/np.pi)

def R(x, y, z, v, Ml, Mu, os):  # показатель рассеяния
    return Y/2*os/(1+s+4*(dw-k*v-dwB(x, y, z, Ml, Mu))**2/Y**2)*s

# константы

mRb = 1.46e-25
h = 6.62e-34
kb = 1.38e-23
Y = 2*np.pi*6.06e6
lam = 780e-9
k = 2*np.pi/lam
mB = 9.274e-24
gu = 2/3
gl = 1/2
g = 9.8 

# параметры симуляции
 
TK = 310 # Kelvin
dw = -13e6 # maybe * 2pi
A = 1600*1e-4 # Tl/m - B grad
TMOL = 10e-3
T = 5e-3
TSIM = 30e-3
s = 10

dt = 1e-8
t = [0]
Nu = np.zeros(7)
Nl = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
vz = [0.01]# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
z = [5e-4]


# силы осцилляторов 
os01 = 2/5
os10 = 1/5
os12 = 2/3
os21 = 1/15
os23 = 1
os22 = 1/3
os11 = 8/15
os00 = 3/5

i = 0
n = 1
while i < int(TSIM/dt):
    for j in range(int(TMOL/dt)):
        
        # населённость нижних
        dNl_2dt =  -(R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0]) + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2])) + Y*(os23*Nu[0]+os22*Nu[1]+os21*Nu[2])
        dNl_1dt =  -(R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])) + Y*(os12*Nu[1]+os11*Nu[2]+os10*Nu[3])
        dNl0dt =  -(R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4])) + Y*(os01*Nu[2]+os00*Nu[3]+os01*Nu[4])
        dNl1dt =  -(R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])) + Y*(os10*Nu[3]+os11*Nu[4]+os12*Nu[5])
        dNl2dt =  -(R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4]) + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])) + Y*(os21*Nu[4]+os22*Nu[5]+os23*Nu[6])

        # населённость верхних
        dNu_3dt = -Y*Nu[0] + R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0])
        dNu_2dt = -Y*Nu[1] + R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1])
        dNu_1dt = -Y*Nu[2] + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2]) + R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2])
        dNu0dt = -Y*Nu[3] + R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])
        dNu1dt = -Y*Nu[4] + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4]) + R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4])
        dNu2dt = -Y*Nu[5] + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])
        dNu3dt = -Y*Nu[6] + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])

        # скорость рассеивания фотонов
        dydt = Y*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 

        # количество рассеяных фотонов
        flg = False
        N = 0
        N += dydt*dt
        if N >= n:
            flg = True

        # ускорение
        az = h/mRb/lam*( R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) 
                          - R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4])
                        + R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) 
                          - R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])
                        + R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4]) 
                          - R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6]) 
                        + R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1]) 
                          - R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])
                        + R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0]) 
                          - R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2]))
        az = az - g
        
        # населённость нижних
        Nl[0] += dNl_2dt*dt
        Nl[1] += dNl_1dt*dt
        Nl[2] += dNl0dt*dt
        Nl[3] += dNl1dt*dt
        Nl[4] += dNl2dt*dt

        # населённость верхних
        Nu[0] += dNu_3dt*dt
        Nu[1] += dNu_2dt*dt
        Nu[2] += dNu_1dt*dt
        Nu[3] += dNu0dt*dt
        Nu[4] += dNu1dt*dt
        Nu[5] += dNu2dt*dt
        Nu[6] += dNu3dt*dt

        print(Nl[0], Nl[1], Nl[2], Nl[3], Nl[4], Nu[0], Nu[1], Nu[2], Nu[3], Nu[4], Nu[5], Nu[6])
        z.append(z[-1]+vz[-1]*dt)
        vz.append(vz[-1] + az*dt)
        t.append(dt*(i+j))

    i += int(TMOL/dt)

print(Nl[0]+Nl[1]+Nl[2]+Nl[3]+Nl[4]+Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6])
t = np.array(t)
vz = np.array(vz)
z = np.array(z)

plt.plot(t*1e3, z*1e3)

plt.show()