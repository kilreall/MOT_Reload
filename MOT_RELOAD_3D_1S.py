import numpy as np 
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.special import erf
from scipy.stats import maxwell

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


def vel_dist():
    # parameters
    v_0 = 200 #np.sqrt(kb*TK/mRb)*1e-3 #km/s
    v_esc = 500 #7*1e-3 #km/s
    N = 10000

    # CDF(v_esc)
    cdf_v_esc = maxwell.cdf(v_esc,scale=v_0/np.sqrt(2))

    # pdf for the distribution
    def f_MB(v_mag):
        n_0 = np.power(np.pi,3/2)*np.square(v_0)*(v_0*erf(v_esc/v_0)-(2/np.sqrt(np.pi))*v_esc*np.exp(-np.square(v_esc/v_0)))
        return (1/n_0)*(4*np.pi*np.square(v_mag))*np.exp(-np.square(v_mag/v_0))*np.heaviside(v_esc-v_mag,0)

    # plot the pdf
    x = np.arange(600)
    y = [f_MB(i) for i in x]
    plt.plot(x,y,label='pdf')

    # use inverse transform sampling to get the truncated Maxwell-Boltzmann distribution
    sample = maxwell.ppf(np.random.rand(N)*cdf_v_esc,scale=v_0/np.sqrt(2))

    # plot the histogram of the samples
    plt.hist(sample,bins=100,histtype='step',density=True,label='histogram')
    plt.xlabel('v_mag')
    plt.legend()
    plt.show()

    return 1



@njit(parallel = True)
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
    return [dv, dr]

@njit
def B(axis, x, y, z): # magnetic field
    if axis == 0 : return -2*z*A
    elif axis == 1 : return x*A
    else : return y*A

@njit
def dwB(axis, x, y, z, Ml, Mu):  # Zeeman shift
    return (gu*Mu-gl*Ml)*B(axis, x, y, z)*mB/(h/2/np.pi)

@njit(parallel = True)
def sd(axis, x, y, z):
    r = np.array([z, x, y])
    r[axis] = 0
    rm2 = np.sum(r**2)
    if rm2 > rl2: return 0
    else:
        return s*np.exp(-2*rm2**2/re2**2)


@njit
def R(axis, x, y, z, v, Ml, Mu, os):  # scattering rate with magnetic field
    return Y/2*os/(1+s+4*(dw-k*v-dwB(axis, x, y, z, Ml, Mu))**2/Y**2)*sd(axis, x, y, z)


def Rrb(x, y, z, v, s):  # scattering rate for retrback beams
    return Y/2/(1+s+4*(dw-k*v)**2/Y**2)*sd(0, x, y, z) #incorrect

@njit(parallel = True)
def simulation(Nl, Nu):
    t = np.arange(0, int(TSIM/dt)+1)*dt
    # Nu = np.zeros(7)
    # Nl = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    
    # coord and speed initialization
    vz = np.zeros(int(TSIM/dt)+1)# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
    vx = np.zeros(int(TSIM/dt)+1)# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
    vy = np.zeros(int(TSIM/dt)+1)# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
    vz[0], vx[0], vy[0] = vz0, vx0, vy0
    z = np.zeros(int(TSIM/dt)+1)
    x = np.zeros(int(TSIM/dt)+1)
    y = np.zeros(int(TSIM/dt)+1)
    z[0], x[0], y[0] = z0, x0, y0

    # n = 1
    # N = 0
    # f1 = rvv(9, 3)
    i = 0
    while i < int(TSIM/dt):

        # MOT Simulation
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
            # az
            az = fk*( R(0, x[i+j], y[i+j], z[i+j], +vz[i+j], 0, -1, os01)*(Nl[2]-Nu[2])
                            - R(0, x[i+j], y[i+j], z[i+j], -vz[i+j], 0, 1, os01)*(Nl[2]-Nu[4])
                            + R(0, x[i+j], y[i+j], z[i+j], +vz[i+j], 1, 0, os10)*(Nl[3]-Nu[3]) 
                            - R(0, x[i+j], y[i+j], z[i+j], -vz[i+j], 1, 2, os12)*(Nl[3]-Nu[5])
                            + R(0, x[i+j], y[i+j], z[i+j], +vz[i+j], 2, 1, os21)*(Nl[4]-Nu[4]) 
                            - R(0, x[i+j], y[i+j], z[i+j], -vz[i+j], 2, 3, os23)*(Nl[4]-Nu[6]) 
                            + R(0, x[i+j], y[i+j], z[i+j], +vz[i+j], -1, -2, os12)*(Nl[1]-Nu[1]) 
                            - R(0, x[i+j], y[i+j], z[i+j], -vz[i+j], -1, 0, os10)*(Nl[1]-Nu[3])
                            + R(0, x[i+j], y[i+j], z[i+j], +vz[i+j], -2, -3, os23)*(Nl[0]-Nu[0]) 
                            - R(0, x[i+j], y[i+j], z[i+j], -vz[i+j], -2, -1, os21)*(Nl[0]-Nu[2]))
            az -= g  # gravity
            # ax
            ax = fk*( R(1, x[i+j], y[i+j], z[i+j], +vx[i+j], 0, 1, os01)*(Nl[2]-Nu[2])
                            - R(1, x[i+j], y[i+j], z[i+j], -vx[i+j], 0, -1, os01)*(Nl[2]-Nu[4])
                            + R(1, x[i+j], y[i+j], z[i+j], +vx[i+j], 1, 2, os10)*(Nl[3]-Nu[3]) 
                            - R(1, x[i+j], y[i+j], z[i+j], -vx[i+j], 1, 0, os12)*(Nl[3]-Nu[5])
                            + R(1, x[i+j], y[i+j], z[i+j], +vx[i+j], 2, 3, os21)*(Nl[4]-Nu[4]) 
                            - R(1, x[i+j], y[i+j], z[i+j], -vx[i+j], 2, 1, os23)*(Nl[4]-Nu[6]) 
                            + R(1, x[i+j], y[i+j], z[i+j], +vx[i+j], -1, 0, os12)*(Nl[1]-Nu[1]) 
                            - R(1, x[i+j], y[i+j], z[i+j], -vx[i+j], -1, -2, os10)*(Nl[1]-Nu[3])
                            + R(1, x[i+j], y[i+j], z[i+j], +vx[i+j], -2, -1, os23)*(Nl[0]-Nu[0]) 
                            - R(1, x[i+j], y[i+j], z[i+j], -vx[i+j], -2, -3, os21)*(Nl[0]-Nu[2]))

            # ay
            ay = fk*( R(2, x[i+j], y[i+j], z[i+j], +vz[i+j], 0, 1, os01)*(Nl[2]-Nu[2])
                            - R(2, x[i+j], y[i+j], z[i+j], -vy[i+j], 0, -1, os01)*(Nl[2]-Nu[4])
                            + R(2, x[i+j], y[i+j], z[i+j], +vy[i+j], 1, 2, os10)*(Nl[3]-Nu[3]) 
                            - R(2, x[i+j], y[i+j], z[i+j], -vy[i+j], 1, 0, os12)*(Nl[3]-Nu[5])
                            + R(2, x[i+j], y[i+j], z[i+j], +vy[i+j], 2, 3, os21)*(Nl[4]-Nu[4]) 
                            - R(2, x[i+j], y[i+j], z[i+j], -vy[i+j], 2, 1, os23)*(Nl[4]-Nu[6]) 
                            + R(2, x[i+j], y[i+j], z[i+j], +vy[i+j], -1, 0, os12)*(Nl[1]-Nu[1]) 
                            - R(2, x[i+j], y[i+j], z[i+j], -vy[i+j], -1, -2, os10)*(Nl[1]-Nu[3])
                            + R(2, x[i+j], y[i+j], z[i+j], +vy[i+j], -2, -1, os23)*(Nl[0]-Nu[0]) 
                            - R(2, x[i+j], y[i+j], z[i+j], -vy[i+j], -2, -3, os21)*(Nl[0]-Nu[2]))


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

            x[i+j+1] = x[i+j]+vx[i+j]*dt
            vx[i+j+1] = vx[i+j] + ax*dt

            y[i+j+1] = y[i+j]+vy[i+j]*dt
            vy[i+j+1] = vy[i+j] + ay*dt

            # simplified photon scattering
            if sd(0, x[i+j], y[i+j], z[i+j]) + sd(1, x[i+j], y[i+j], z[i+j]) + sd(2, x[i+j], y[i+j], z[i+j]) != 0:
                dvdr = rvvv(n_ph, dtt, sc_c)

                z[i+j+1] = z[i+j+1] + dvdr[1][2]
                vz[i+j+1] = vz[i+j+1] + dvdr[0][2]

                x[i+j+1] = x[i+j+1] + dvdr[1][0]
                vx[i+j+1] = vx[i+j+1] + dvdr[0][0]

                y[i+j+1] = y[i+j+1] + dvdr[1][1]
                vy[i+j+1] = vy[i+j+1] + dvdr[0][1]
  
            print(dt*(i+j)/TSIM*100, "%")
            j +=1
        i += j
        print(dt*(i+j)/TSIM*100, "%")
        
        # subdopler cooling
        j = 0
        while j < int(TSD/dt) and i+j < int(TSIM/dt):
            
            #acceleration
            # az
            if sd(0, x[i+j], y[i+j], z[i+j]) != 0:
                az = ap*dws*Y/((dws)**2+Y**2/4)*vz[i+j] # subdopller cooling
            az -= g # gravity

            # ax
            if sd(1, x[i+j], y[i+j], z[i+j]) != 0:
                ax = ap*dws*Y/((dws)**2+Y**2/4)*vx[i+j] # subdopller cooling

            # ay
            if sd(2, x[i+j], y[i+j], z[i+j]) != 0:
                ay = ap*dws*Y/((dws)**2+Y**2/4)*vy[i+j] # subdopller cooling

            # new speed and coordinate
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] + az*dt

            x[i+j+1] = x[i+j]+vx[i+j]*dt
            vx[i+j+1] = vx[i+j] + ax*dt

            y[i+j+1] = y[i+j]+vy[i+j]*dt
            vy[i+j+1] = vy[i+j] + ay*dt

            # simplified photon scattering
            if sd(0, x[i+j], y[i+j], z[i+j]) + sd(1, x[i+j], y[i+j], z[i+j]) + sd(2, x[i+j], y[i+j], z[i+j]) != 0:
                dvdr = rvvv(n_ph, dtt, sc_c*sc_cDS)

                z[i+j+1] = z[i+j+1] + dvdr[1][2]
                vz[i+j+1] = vz[i+j+1] + dvdr[0][2]

                x[i+j+1] = x[i+j+1] + dvdr[1][0]
                vx[i+j+1] = vx[i+j+1] + dvdr[0][0]

                y[i+j+1] = y[i+j+1] + dvdr[1][1]
                vy[i+j+1] = vy[i+j+1] + dvdr[0][1]

            print(dt*(i+j)/TSIM*100, "%")
            j += 1
        i += j
        print(dt*(i+j)/TSIM*100, "%")

        # fall
        j = 0
        while j < int(2*T/dt) and i+j < int(TSIM/dt):

            # new v and r
            z[i+j+1] = z[i+j]+vz[i+j]*dt
            vz[i+j+1] = vz[i+j] - g*dt

            x[i+j+1] = x[i+j]+vx[i+j]*dt
            vx[i+j+1] = vx[i+j]

            y[i+j+1] = y[i+j]+vy[i+j]*dt
            vy[i+j+1] = vy[i+j]

            print(dt*(i+j)/TSIM*100, "%")
            j += 1
        i += j
        print(dt*(i+j)/TSIM*100, "%")

        # # return back
        # j = 0
        # while j < int(TRB/dt) and i+j < int(TSIM/dt):
        #     #accelaration due to light force
        #     az = fk*( Rrb(0, 0, z[i+j], +vz[i+j], s1) 
        #                     - Rrb(0, 0, z[i+j], -vz[i+j], s2))
        #     az -= g # gravity

        #     # new speed and coordinate
        #     z[i+j+1] = z[i+j]+vz[i+j]*dt
        #     vz[i+j+1] = vz[i+j] + az*dt

        #     # simplified photon scattering
        #     dvzdz = rvvv(n_ph, dtt, sc_c)
        #     z[i+j+1] = z[i+j+1] + dvzdz[1]
        #     vz[i+j+1] = vz[i+j+1] + dvzdz[0]

        #     j += 1
        #     print(dt*(i+j)/TSIM*100, "%")
        # i += j
        # print(dt*(i+j)/TSIM*100, "%")


    # separation
    vz = vz[:i+j+1]
    z = z[:i+j+1]
    t = t[:i+j+1]
    return t, vz, vx, vy, z, x, y

def n_atom_sim(N):
    vx = np.random
    return 1

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
ap = h/2/np.pi*k**2/2*cJ/mRb/3 #/3-3D constant for force/acceleration calculations for subdoppler cooling

# simulation parameters

N = 1
TK = 310 # Kelvin
dw = -Y/2 # doppler detuning 
dws = -3*Y # subdoppler detuning
A = 1600*1e-4 # Tl/m - B grad
TMOL = 40e-3
TSD = 40e-3 # subdoppler time
T = 15e-3
TRB = 4.17e-3 # atoms MOT reload time
TSIM = 200e-3 
s = 10 # saturation peak r = 0
rl2 = 2e-3**2 # laser beam radius ^2
re2 = 1e-3 # laser beam 1/e^2 radius ^2
s1, s2 = 3, 2
vz0, vx0, vy0 = 1, 1, 1
z0, x0, y0 = 5e-4, 5e-4, 5e-4
Nl = np.array([0.25570579682352823, 0.07815520905720487, 0.078038171580307, 0.07824747837591599, 0.2565783535707158])
Nu = np.array([0.08249243609654235, 0.018835495263264144, 0.01881647571398126, 0.012523392863259803, 0.018845918641934044, 0.018882228353764675, 0.08287904365948709])

# photon scattering
dydt = Y*np.sum(Nu) # photon scattering rate
dtt = 1/dydt # time  for one photon scattering ~ 10-7
sc_c = fk*Y*np.sum(Nu) # scattering const before rand vec
sc_cDS = 1/3 # scatt const fix for subdoppler cooling
n_ph = 400 # amount of scattering photons per step; 10-100 for fast count
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




# t, vz, vx, vy, z, x, y = simulation(Nl, Nu)
vel_dist()

# print(Nl[0]+Nl[1]+Nl[2]+Nl[3]+Nl[4]+Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6])

plt.figure(1)
plt.plot(t*1e3, vz, label = "vz")
plt.plot(t*1e3, vx+1, label = "vx")
plt.plot(t*1e3, vy+2, label = "vy")
plt.title("speed")
plt.xlabel("t, [ms]")
plt.ylabel("vp, [m/s]")

plt.figure(2)
plt.plot(t*1e3, z*1e3, label = "z")
plt.plot(t*1e3, x*1e3, label = "x")
plt.plot(t*1e3, y*1e3, label = "y")
plt.title("coordinate")
plt.xlabel("t, [ms]")
plt.ylabel("r, [mm]")

# print(h*k/mRb/2/np.pi*n_ph, 'm/s') # speed change due to n_ph
avv = vz[int(10e-3/dt):int(39e-3/dt)]
avv = np.average(avv**2)
print(avv*mRb/kb*1e6, "Tz D, [mkK]")

avv = vz[int(41e-3/dt):int(79e-3/dt)]
avv = np.average(avv**2)
print(avv*mRb/kb*1e6, "Tz SD, [mkK]")

plt.legend()
plt.show()