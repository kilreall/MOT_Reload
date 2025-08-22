import numpy as np 
import matplotlib.pyplot as plt
from numba import njit, prange
##3

def rvv(m, n): # m случайных единичных векторов в n-мерном пространстве и их сумма медленно
    vectors = np.empty((m, n))
    for i in range(m):
        v = np.random.normal(size=n)  # Генерация из нормального распределения
        norm = np.sqrt(np.sum(v**2))  # Нормализация
        vectors[i] = v / norm
    return np.sum(vectors, axis=0)



def rv(): # cлучайнный единичный вектор
    """Генерирует случайный единичный вектор на n-мерной сфере."""
    v = np.random.normal(size=3)  # Генерация из стандартного нормального распределения
    norm = np.linalg.norm(v)     # Вычисление длины вектора
    if norm == 0:                # На случай, если все компоненты нулевые (крайне маловероятно)
        return rv(3)
    return v / norm             # Нормализация до единичной длины


@njit
def rvvv(n_ph, dtt, sc_c): # counting changing of v and r because of photon scattering
    vec = np.random.normal(0, 1, (n_ph, 3))
    sum_of_squares = np.sum(vec ** 2, axis=1)
    norms = np.sqrt(sum_of_squares)
    vec = vec / norms.reshape(-1, 1)
    da = vec*sc_c
    dv = da*dtt
    dr = dv*dtt
    dv = np.sum(dv, axis=0)
    dr = np.sum(dr, axis=0)
    return [dv[2], dr[2]]

def sid():
    return 1

@njit
def B(x, y, z): # magnetic field
    return -2*z*A

@njit
def dwB(x, y, z, Ml, Mu):  # Zeeman shift
    return (gu*Mu-gl*Ml)*B(x, y, z)*mB/(h/2/np.pi)

@njit
def R(x, y, z, v, Ml, Mu, os):  # scattering rate with magnetic field
    return Y/2*os/(1+s+4*(dw-k*v-dwB(x, y, z, Ml, Mu))**2/Y**2)*s

@njit
def Rrb(x, y, z, v, s):  # scattering rate without magnetic field
    return Y/2/(1+s+4*(dw-k*v)**2/Y**2)*s


def simulation(Nl, Nu):
    t = np.arange(0, int(TSIM/dt)+1)*dt
    # Nu = np.zeros(7)
    # Nl = np.array([1/5, 1/5, 1/5, 1/5, 1/5])

    vz = np.zeros(int(TSIM/dt)+1)# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
    vz[0] = vz0
    z = np.zeros(int(TSIM/dt)+1)
    z[0] = z0

    # n = 1
    # N = 0
    # f1 = rvv(9, 3)
    i = 0
    while i < int(TSIM/dt):
        j = 0
        while j < int(TMOL/dt) and i+j < int(TSIM/dt):  # MOT

            # down population
            # dNl_2dt =  -(R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0]) + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2])) + Y*(os23*Nu[0]+os22*Nu[1]+os21*Nu[2])
            # dNl_1dt =  -(R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])) + Y*(os12*Nu[1]+os11*Nu[2]+os10*Nu[3])
            # dNl0dt =  -(R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4])) + Y*(os01*Nu[2]+os00*Nu[3]+os01*Nu[4])
            # dNl1dt =  -(R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])) + Y*(os10*Nu[3]+os11*Nu[4]+os12*Nu[5])
            # dNl2dt =  -(R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4]) + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])) + Y*(os21*Nu[4]+os22*Nu[5]+os23*Nu[6])

            # up population
            # dNu_3dt = -Y*Nu[0] + R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0])
            # dNu_2dt = -Y*Nu[1] + R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1])
            # dNu_1dt = -Y*Nu[2] + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2]) + R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2])
            # dNu0dt = -Y*Nu[3] + R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])
            # dNu1dt = -Y*Nu[4] + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4]) + R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4])
            # dNu2dt = -Y*Nu[5] + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])
            # dNu3dt = -Y*Nu[6] + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])

            # photon scattering rate slow
            # dydt = Y*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 


            # amount of scattered photons slow
            # N += dydt*dt
            # if N >= n:
            #     f1 = rv()
            #     n += 1

            # acceleration
            # due to light force
            az = fk*( R(0, 0, z[i+j], +vz[i+j], 0, -1, os01)*(Nl[2]-Nu[2])
                            - R(0, 0, z[i+j], -vz[i+j], 0, 1, os01)*(Nl[2]-Nu[4])
                            + R(0, 0, z[i+j], +vz[i+j], 1, 0, os10)*(Nl[3]-Nu[3]) 
                            - R(0, 0, z[i+j], -vz[i+j], 1, 2, os12)*(Nl[3]-Nu[5])
                            + R(0, 0, z[i+j], +vz[i+j], 2, 1, os21)*(Nl[4]-Nu[4]) 
                            - R(0, 0, z[i+j], -vz[i+j], 2, 3, os23)*(Nl[4]-Nu[6]) 
                            + R(0, 0, z[i+j], +vz[i+j], -1, -2, os12)*(Nl[1]-Nu[1]) 
                            - R(0, 0, z[i+j], -vz[i+j], -1, 0, os10)*(Nl[1]-Nu[3])
                            + R(0, 0, z[i+j], +vz[i+j], -2, -3, os23)*(Nl[0]-Nu[0]) 
                            - R(0, 0, z[i+j], -vz[i+j], -2, -1, os21)*(Nl[0]-Nu[2]))
            az -= g  # gravity

            # photon scattering slow
            # f1 = rvv(rvvv(9, 3))
            # az += fk*Y*f1[2]*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 

            
            # # ground population
            # Nl[0] += dNl_2dt*dt
            # Nl[1] += dNl_1dt*dt
            # Nl[2] += dNl0dt*dt
            # Nl[3] += dNl1dt*dt
            # Nl[4] += dNl2dt*dt

            # # up population
            # Nu[0] += dNu_3dt*dt
            # Nu[1] += dNu_2dt*dt
            # Nu[2] += dNu_1dt*dt
            # Nu[3] += dNu0dt*dt
            # Nu[4] += dNu1dt*dt
            # Nu[5] += dNu2dt*dt
            # Nu[6] += dNu3dt*dt

            # print(Nl[0], Nl[1], Nl[2], Nl[3], Nl[4], Nu[0], Nu[1], Nu[2], Nu[3], Nu[4], Nu[5], Nu[6])

            # new speed and coordinate
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] + az*dt

            # simplified photon scattering
            dvzdz = rvvv(n_ph, dtt, sc_c)
            z[i+j+1] = z[i+j+1] + dvzdz[1]
            vz[i+j+1] = vz[i+j+1] + dvzdz[0]

  
            print(dt*(i+j)/TSIM*100, "%")
            j +=1
        i += j
        print(dt*(i+j)/TSIM*100, "%")
        
        # subdopler cooling
        j = 0
        while j < int(TSD/dt) and i+j < int(TSIM/dt):
            az = ap*dws*Y/((dws)**2+Y**2/4)*vz[i+j] # subdopller cooling
            az -= g # gravity

            # new speed and coordinate
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] + az*dt

            # simplified photon scattering
            dvzdz = rvvv(n_ph, dtt, sc_c)
            z[i+j+1] = z[i+j+1] + dvzdz[1]
            vz[i+j+1] = vz[i+j+1] + dvzdz[0]
            print(dt*(i+j)/TSIM*100, "%")
            j += 1
        i += j
        print(dt*(i+j)/TSIM*100, "%")

        # fall
        j = 0
        while j < int(2*T/dt) and i+j < int(TSIM/dt):
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] - g*dt
            print(dt*(i+j)/TSIM*100, "%")
            j += 1
        i += j
        print(dt*(i+j)/TSIM*100, "%")

        # return back
        j = 0
        while j < int(TRB/dt) and i+j < int(TSIM/dt):
            #accelaration due to light force
            az = fk*( Rrb(0, 0, z[i+j], +vz[i+j], s1) 
                            - Rrb(0, 0, z[i+j], -vz[i+j], s2))
            az -= g # gravity

            # new speed and coordinate
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] + az*dt

            # simplified photon scattering
            dvzdz = rvvv(n_ph, dtt, sc_c)
            z[i+j+1] = z[i+j+1] + dvzdz[1]
            vz[i+j+1] = vz[i+j+1] + dvzdz[0]

            j += 1
            print(dt*(i+j)/TSIM*100, "%")
        i += j
        print(dt*(i+j)/TSIM*100, "%")


    # separation
    vz = vz[:i+j+1]
    z = z[:i+j+1]
    t = t[:i+j+1]
    return t, vz, z

# constants

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
fk = h/mRb/lam # constant for force/acceleration calculations for doppler cooling
cJ = 11 # subdopler cooling constand for 2-3' 
ap = h/2/np.pi*k**2/2*cJ/mRb # constant for force/acceleration calculations subfor doppler cooling

# simulation parameters
 
TK = 310 # Kelvin
dw = -Y/2 # doppler detuning maybe * 2pi
dws = -3*Y # subdoppler detuning
A = 1600*1e-4 # Tl/m - B grad
TMOL = 10e-3
TSD = 10e-3 # subdoppler time
T = 15e-3
TRB = 4.17e-3
TSIM = 300e-3
s = 10 # saturation koef
s1, s2 = 3, 2
vz0 = 1
z0 = 1e-3
Nl = np.array([0.26153395045812866, 0.07873626045853599, 0.07827201728294596, 0.07872747249646815, 0.2614468287734205])
Nu = np.array([0.07942201429289736, 0.017735310701816314, 0.01766200744437471, 0.011693191684085063, 0.017658922501618563, 0.017730432180398364, 0.07938159172507568])

# photon scattering
dydt = Y*np.sum(Nu) # photon scattering rate
dtt = 1/dydt # time  for one photon scattering ~ 10-7
sc_c = fk*Y*np.sum(Nu) # scattering const before rand vec
n_ph = 100 # amount of scattering photons per step; 10-100 for fast count
dt = dtt*n_ph # time step n_ph = 100 ~ 1e-5
# dt = 1e-6 without photon scattering fast 1e-8 slow


# oscillator strength
os01 = 2/5
os10 = 1/5
os12 = 2/3
os21 = 1/15
os23 = 1
os22 = 1/3
os11 = 8/15
os00 = 3/5




t, vz, z = simulation(Nl, Nu)

# print(Nl[0]+Nl[1]+Nl[2]+Nl[3]+Nl[4]+Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6])

plt.figure(1)
plt.plot(t*1e3, vz)
plt.title("speed")
plt.xlabel("t, [ms]")
plt.ylabel("vz, [m/s]")

plt.figure(2)
plt.plot(t*1e3, z*1e3)
plt.title("coordinate")
plt.xlabel("t, [ms]")
plt.ylabel("z, [mm]")

# print(h*k/mRb/2/np.pi*n_ph, 'm/s') # speed change due to n_ph
avv = vz[int(10e-3/dt):int(39e-3/dt)]
avv = np.average(avv**2)
print(avv*mRb/kb*1e6)

avv = vz[int(41e-3/dt):int(79e-3/dt)]
avv = np.average(avv**2)
print(avv*mRb/kb*1e6)

plt.show()