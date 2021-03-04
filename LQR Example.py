import numpy as np
import control as ctl
import control.matlab as matlab
import matplotlib.pyplot as plt
from time import time


A = np.array([[0,1,0],[0,0,1],[-1,-5,-6]])
B = np.array([[0],[0],[1]])

print("A = \n", A)
print("B = \n", B)

Q = np.array([[50000,0,0],[0,1000,0],[0,0,10]])
R = np.array([[1]])
K, S, E = ctl.lqr(A, B, Q, R)
print("K = \n", K)

t0 = 0
t_end = 5
dt = 0.01
t = np.arange(t0,t_end+dt,dt)     # deret waktu 0 - 5 dengan kenaikan dt (0,01)

tolX1 = 0.10 #toleransi x1
rt = t_end
st = t_end
rtTarget = 0.2

sistem = ctl.ss((A-B*K),np.identity(3),np.identity(3),np.identity(3))

x = matlab.initial(sistem,t,np.array([1,0,0]))
x1 = [1,0,0]@np.transpose(x[0])
x2 = [0,1,0]@np.transpose(x[0])
x3 = [0,0,1]@np.transpose(x[0])



OvS = 0
peakOv = np.zeros(30)
OvKe = 0
OvBef = 0 
J = 0
t0 = int(round(time() * 1000))

for i in range(0, len(t)):
#    print("data ke-",i," = ",x1[i])
#    print("waktu ke-",i," = ",t[i])
    if (abs(x1[i]) < tolX1 and rt == t_end):
        rt = t[i]
        st = t[i]
    if (t[i] > rt and abs(x1[i]) > tolX1):
        OvS = x1[i]
        if (abs(OvS) < abs(OvBef) ):
            peakOv[OvKe] = OvBef
        else:
            OvBef = OvS
    if (t[i] > rt and peakOv[OvKe] > 0):
        OvKe = OvKe +1
        OvS = 0
        OvBef = 0
    
    #menghitung cost function
    t1 = time()*1000
    dt = t1 - t0
#    print("t0 = ",t0/1000)
#    print("t1 = ",t1/1000)
#    print("dt = ",dt/1000)
    t0 = t1
    xCF = np.array([[x1[i]],[x2[i]],[x3[i]]])
    u = -K@xCF
#    print("u = ", u)
    J = J + ((np.transpose(xCF)@Q@xCF) + (np.transpose(u)@R@u))*dt/1000
#    print("J ke-",i," = ",J)


#cost funtion

print("Rise Time = ",rt)
print("Overshoot =\n",peakOv)
print("Cost Function = ",J)

fig, ax=plt.subplots()
ax.plot(t,x1)
ax.set_ylabel('$x_1$',fontsize=12)
ax.set_xlabel('$t$',fontsize=12)
ax.grid()

fig,ax=plt.subplots()
ax.plot(t,x2)
ax.set_ylabel('$x_2$',fontsize=12)
ax.set_xlabel('$t$',fontsize=12)
ax.grid()

fig,ax=plt.subplots()
ax.plot(t,x3)
ax.set_ylabel('$x_3$',fontsize=12)
ax.set_xlabel('$t$',fontsize=12)
ax.grid()