import numpy as np 
import matplotlib.pyplot as plt
from numba import njit, prange


def rvvv(m, n):
    """Генерирует m случайных единичных векторов в n-мерном пространстве."""
    vectors = np.empty((m, n))
    for i in range(m):
        v = np.random.normal(size=n)  # Генерация из нормального распределения
        norm = np.sqrt(np.sum(v**2))  # Нормализация
        vectors[i] = v / norm
    return vectors


def rvv(vectors):
    """Суммирует все векторы."""
    return np.sum(vectors, axis=0)


def rv():
    """Генерирует случайный единичный вектор на n-мерной сфере."""
    v = np.random.normal(size=3)  # Генерация из стандартного нормального распределения
    norm = np.linalg.norm(v)     # Вычисление длины вектора
    if norm == 0:                # На случай, если все компоненты нулевые (крайне маловероятно)
        return rv(3)
    return v / norm             # Нормализация до единичной длины

@njit
def B(x, y, z): # маг поле
    return -2*z*A

@njit
def dwB(x, y, z, Ml, Mu):  # зеемановское смещение
    return (gu*Mu-gl*Ml)*B(x, y, z)*mB/(h/2/np.pi)

@njit
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
TSD = 10e-3 # subdoppler time
T = 5e-3
TSIM = 30e-3
s = 10

dt = 1e-6
t = [0]
Nu = np.zeros(7)
Nl = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
vz = [1]# [np.random.normal(0, np.sqrt(kb*TK/mRb))]
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

Nl = np.array([0.26153395045812866, 0.07873626045853599, 0.07827201728294596, 0.07872747249646815, 0.2614468287734205])
Nu = np.array([0.07942201429289736, 0.017735310701816314, 0.01766200744437471, 0.011693191684085063, 0.017658922501618563, 0.017730432180398364, 0.07938159172507568])
dydt = Y*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 


def simulation():
    fk = h/mRb/lam # константа для вычисления силы
    i = 0
    n = 1
    # N = 0
    # f1 = rvv(rvvv(9, 3))
    while i < int(TSIM/dt):
        j = 0
        while j < int(TMOL/dt) and i+j < int(TSIM/dt):
            # населённость нижних
            # dNl_2dt =  -(R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0]) + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2])) + Y*(os23*Nu[0]+os22*Nu[1]+os21*Nu[2])
            # dNl_1dt =  -(R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])) + Y*(os12*Nu[1]+os11*Nu[2]+os10*Nu[3])
            # dNl0dt =  -(R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4])) + Y*(os01*Nu[2]+os00*Nu[3]+os01*Nu[4])
            # dNl1dt =  -(R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])) + Y*(os10*Nu[3]+os11*Nu[4]+os12*Nu[5])
            # dNl2dt =  -(R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4]) + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])) + Y*(os21*Nu[4]+os22*Nu[5]+os23*Nu[6])

            # населённость верхних
            # dNu_3dt = -Y*Nu[0] + R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0])
            # dNu_2dt = -Y*Nu[1] + R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1])
            # dNu_1dt = -Y*Nu[2] + R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2]) + R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2])
            # dNu0dt = -Y*Nu[3] + R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) + R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])
            # dNu1dt = -Y*Nu[4] + R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4]) + R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4])
            # dNu2dt = -Y*Nu[5] + R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])
            # dNu3dt = -Y*Nu[6] + R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6])

            # скорость рассеивания фотонов
            # dydt = Y*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 


            # количество рассеяных фотонов
            # N += dydt*dt
            # if N >= n:
            #     f1 = rv()
            #     n += 1

            # ускорение
            az = fk*( R(0, 0, z[-1], +vz[-1], 0, -1, os01)*(Nl[2]-Nu[2]) 
                            - R(0, 0, z[-1], -vz[-1], 0, 1, os01)*(Nl[2]-Nu[4])
                            + R(0, 0, z[-1], +vz[-1], 1, 0, os10)*(Nl[3]-Nu[3]) 
                            - R(0, 0, z[-1], -vz[-1], 1, 2, os12)*(Nl[3]-Nu[5])
                            + R(0, 0, z[-1], +vz[-1], 2, 1, os21)*(Nl[4]-Nu[4]) 
                            - R(0, 0, z[-1], -vz[-1], 2, 3, os23)*(Nl[4]-Nu[6]) 
                            + R(0, 0, z[-1], +vz[-1], -1, -2, os12)*(Nl[1]-Nu[1]) 
                            - R(0, 0, z[-1], -vz[-1], -1, 0, os10)*(Nl[1]-Nu[3])
                            + R(0, 0, z[-1], +vz[-1], -2, -3, os23)*(Nl[0]-Nu[0]) 
                            - R(0, 0, z[-1], -vz[-1], -2, -1, os21)*(Nl[0]-Nu[2]))
            az -= g

            # рассеяние фотонов
            # f1 = rvv(rvvv(9, 3))
            # az += h*Y/mRb/lam*f1[2]*(Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6]) 

            
            # # населённость нижних
            # Nl[0] += dNl_2dt*dt
            # Nl[1] += dNl_1dt*dt
            # Nl[2] += dNl0dt*dt
            # Nl[3] += dNl1dt*dt
            # Nl[4] += dNl2dt*dt

            # # населённость верхних
            # Nu[0] += dNu_3dt*dt
            # Nu[1] += dNu_2dt*dt
            # Nu[2] += dNu_1dt*dt
            # Nu[3] += dNu0dt*dt
            # Nu[4] += dNu1dt*dt
            # Nu[5] += dNu2dt*dt
            # Nu[6] += dNu3dt*dt

            # print(Nl[0], Nl[1], Nl[2], Nl[3], Nl[4], Nu[0], Nu[1], Nu[2], Nu[3], Nu[4], Nu[5], Nu[6])
            z.append(z[-1]+vz[-1]*dt)
            vz.append(vz[-1] + az*dt)
            t.append(dt*(i+j))
            print(dt*(i+j)/TSIM*100, "%")
            j +=1
        i += int(TMOL/dt)
        print(dt*(i+j)/TSIM*100, "%")

        # падение
        j = 0
        while j < int(2*T/dt) and i+j < int(TSIM/dt):
            z.append(z[-1]+vz[-1]*dt)
            vz.append(vz[-1] - g*dt)
            t.append(dt*(i+j))
            print(dt*(i+j)/TSIM*100, "%")
            j += 1
        i += int(2*T/dt) 
        print(dt*(i+j)/TSIM*100, "%")
    return 1

simulation()

# print(Nl[0]+Nl[1]+Nl[2]+Nl[3]+Nl[4]+Nu[0]+Nu[1]+Nu[2]+Nu[3]+Nu[4]+Nu[5]+Nu[6])
t = np.array(t)
vz = np.array(vz)
z = np.array(z)

plt.plot(t*1e3, z*1e3)

print(dydt)

plt.show()