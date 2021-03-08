# SIMULASI SISTEM KENDALI LQR 
# Penelitian Skripsi L1 Controller Navigation
# M Alvi Nurhidayah

import numpy as np
import math
import control as ctl
import control.matlab as matlab
import matplotlib.pyplot as plt
from time import time

# =====================> Fixedwing Configuration
M_body = 1.276
p_body = 0.1
l_body = 0.94
t_body = 0.165
x_body = 0.0
y_body = 0.0
z_body = 0.0
M_sayap = 0.38
p_sayap = 1.435
l_sayap = 0.22
t_sayap = 0.025
x_sayap = 0.045
y_sayap = 0.0
z_sayap = 0.0725
M_Hstab = 0.079
p_Hstab = 0.56
l_Hstab = 0.185
t_Hstab = 0.015
x_Hstab = 0.5675
y_Hstab = 0.0
z_Hstab = 0.0775
M_Vstab = 0.02
p_Vstab = 0.005
l_Vstab = 0.332
t_Vstab = 0.227
x_Vstab = 0.0494
y_Vstab = 0.0
z_Vstab = 0.0285
M_motor = 0.108
r_motor = 0.014
t_motor = 0.014
x_motor = 0.295
y_motor = 0
z_motor = 0.085
M_prop = 0.015
r_prop = 0.254 
t_prop = 0.127
x_prop = 0.295
y_prop = 0.0
z_prop = 0.085
M_pesawat = 2.78

# =====================> Calculating Momen Inertia
Ixx = (((M_body * (l_body*l_body + t_body*t_body) / 12) + (M_body * (y_body*y_body + z_body*z_body)))
    + ((M_sayap * (l_sayap*l_sayap + t_sayap*t_sayap) / 12) + (M_sayap * (y_sayap*y_sayap + z_sayap*z_sayap)))
    + ((M_Hstab * (l_Hstab*l_Hstab + t_Hstab*t_Hstab) / 12) + (M_Hstab * (y_Hstab*y_Hstab + z_Hstab*z_Hstab)))
    + ((M_Vstab * (l_Vstab*l_Vstab + t_Vstab*t_Vstab) / 12) + (M_Vstab * (y_Vstab*y_Hstab + z_Vstab*z_Vstab)))
    + ((M_motor * (3*r_motor*r_motor + t_motor*t_motor) / 12) + (M_motor * (y_motor*y_motor + z_motor*z_motor)))
    + ((M_prop * (3*r_prop*r_prop + t_prop*t_prop) / 12) + (M_prop * (y_prop*y_prop + z_prop*z_prop))))

Iyy = (((M_body * (p_body*p_body + t_body*t_body) / 12) + (M_body * (x_body*x_body + z_body*z_body)))
    + ((M_sayap * (p_sayap*p_sayap + t_sayap*t_sayap) / 12) + (M_sayap * (x_sayap*x_sayap + z_sayap*z_sayap)))
    + ((M_Hstab * (p_Hstab*p_Hstab + t_Hstab*t_Hstab) / 12) + (M_Hstab * (x_Hstab*x_Hstab + z_Hstab*z_Hstab)))
    + ((M_Vstab * (p_Vstab*p_Vstab + t_Vstab*t_Vstab) / 12) + (M_Vstab * (x_Vstab*x_Hstab + z_Vstab*z_Vstab)))
    + ((M_motor * (3*r_motor*r_motor + t_motor*t_motor) / 12) + (M_motor * (x_motor*x_motor + z_motor*z_motor)))
    + ((M_prop * (3*r_prop*r_prop + t_prop*t_prop) / 12) + (M_prop * (x_prop*x_prop + z_prop*z_prop))))

Izz = (((M_body * (p_body*p_body + l_body*l_body) / 12) + (M_body * (x_body*x_body + y_body*y_body)))
    + ((M_sayap * (p_sayap*p_sayap + l_sayap*l_sayap) / 12) + (M_sayap * (x_sayap*x_sayap + y_sayap*y_sayap)))
    + ((M_Hstab * (p_Hstab*p_Hstab + l_Hstab*l_Hstab) / 12) + (M_Hstab * (x_Hstab*x_Hstab + y_Hstab*y_Hstab)))
    + ((M_Vstab * (p_Vstab*p_Vstab + t_Vstab*t_Vstab) / 12) + (M_Vstab * (x_Vstab*x_Hstab + y_Vstab*y_Vstab)))
    + ((M_motor * (r_motor*r_motor) / 2) + (M_motor * (x_motor*x_motor + y_motor*y_motor)))
    + ((M_prop * (r_prop*r_prop) / 2) + (M_prop * (x_prop*x_prop + y_prop*y_prop))))

# Ixx = 0.103557335
# Iyy = 0.129538628
# Izz = 0.219640939
# Tampilkan nilai inersia
print("inersia X : {:.3f}".format(Ixx))
print("inersia Y : {:.3f}".format(Iyy))
print("inersia Z : {:.3f}\n".format(Izz))

# =====================> State Space Modelling
# L1  = 1.435;%panjang sayap yang sejajar dengan horzontal stab (dari CG).
# L2  = 0.66;% Panjang badan (dari CG) sampai ujung ekor pesawat

# Nilai state yang menjadi variabel dalam matrix A dianggap kecil
r = 0.000001
p = 0.000001
q = 0.000001
# Tampilkan nilai r, p, q
print("nilai r : {:.3f}".format(r))
print("nilai p : {:.3f}".format(p))
print("nilai q : {:.3f}\n".format(q))

# State Space Modelling
A = np.array([[0,        1,         0,         0,         0,       0],   # Roll
              [0,        0,         0, (Iyy-Izz)*r/Ixx,   0,       0],   # Kec. Sudut Roll
              [0,        0,         0,         1,         0,       0],   # Pitch
              [0, (Izz-Ixx)*r/Iyy,  0,         0,         0,       0],   # Kec. Sudut Pitch
              [0,        0,         0,         0,         0,       1],   # Yaw
              [0, (Ixx-Iyy)*q/Izz,  0,         0,         0,       0]])  # Kec. Sudut Yaw
eigen_value = np.linalg.eigvals(A)
print("A = \n{}\n".format(np.around(A, 4)))

B = np.array([[0,       0,       0],
              [1/Ixx,   0,       0],
              [0,       0,       0],
              [0,     1/Iyy,     0],
              [0,       0,       0],
              [0,       0,   1/Izz]])
print("B = \n{}\n".format(np.around(B, 4)))

# Eigen Value of A and Controllability
print("EigenValue before ClosedLoop= \n{}\n".format(eigen_value))
controllability = ctl.ctrb(A, B)
controllability = np.linalg.matrix_rank(controllability)
print("controllability : {}\n".format(controllability)) 

C = np.array([[ 1,   0,   0,   0,   0,   0,],
              [ 0,   0,   0,   0,   0,   0,],
              [ 0,   0,   1,   0,   0,   0,],
              [ 0,   0,   0,   0,   0,   0,],
              [ 0,   0,   0,   0,   1,   0,],
              [ 0,   0,   0,   0,   0,   0,]])
print("C = \n{}\n".format(np.around(C, 4)))

D = np.zeros((6,3))

# =====================> Sistem Kendali LQR     
# Nilai untuk ditelaa agar mendapatkan respon sistem yang sesuai
Q = np.array([[ 27,   0,   0,   0,   0,   0],     #Roll
              [ 0,   1,   0,   0,   0,   0],     #gyro roll
              [ 0,   0,   1,   0,   0,   0],     #pitch
              [ 0,   0,   0,   1,   0,   0],     #gyro pitch
              [ 0,   0,   0,   0,   1,   0],     #yaw
              [ 0,   0,   0,   0,   0,   1]])    #gyro yaw
print("Q = \n{}\n".format(np.around(Q, 4)))

# Matriks R mengikuti jumlah u
R = np.array([[ 1,   0,   0],
              [ 0,   1,   0],
              [ 0,   0,   1]])
print("R = \n{}\n".format(np.around(A, 4)))

K, S, E = ctl.lqr(A, B, Q, R)
print("K = \n{}\n".format(np.round(K, 4)))
print("Eigen Value ClosedLoop = \n{}\n".format(E))

# ======================> Simulation
t_start = 0
t_end = 10
dt = 0.01
t = np.arange(t_start,t_end+dt,dt)     # deret waktu 0 - 5 dengan kenaikan dt (0,01)

Acl = A-B*K
Bcl = np.identity(6)
Ccl = np.identity(6)
Dcl = np.zeros((6,6))

sys_cl = ctl.ss((Acl), B, C, D)
sys_ol = ctl.ss(A, B, C, D)

# Inisialisasi Gangguan Awal
roll = math.radians(45) 
gyro_roll = math.radians(0)
pitch = math.radians(45)
gyro_pitch = math.radians(0)
yaw = math.radians(45)
gyro_yaw = math.radians(0)

x = matlab.initial(sys_cl,t,np.array([roll, gyro_roll, pitch, gyro_pitch, yaw, gyro_yaw]))
x1 = [1,0,0,0,0,0]@np.transpose(x[0])
x2 = [0,1,0,0,0,0]@np.transpose(x[0])
x3 = [0,0,1,0,0,0]@np.transpose(x[0])
x4 = [0,0,0,1,0,0]@np.transpose(x[0])
x5 = [0,0,0,0,1,0]@np.transpose(x[0])
x6 = [0,0,0,0,0,1]@np.transpose(x[0])

print("roll awal \t: ", x1[0])
print("gyro roll awal \t: ", x2[0])
print("pitch awal \t: ", x3[0])
print("gyro pitch awal : ", x4[0])
print("yaw awal \t: ", x5[0])
print("gyro yaw awal \t:  {}\n".format(x6[0]))

# Initialize Desired Control Paremeter
toleransi_sudut = 0.1
toleransi_kecsudut = 0.1
rt = t_end
st = t_end
rtTarget = 0.2

OvS = 0
peakOv = np.zeros(30)
OvKe = 0
OvBef = 0 
J = 0
t0 = int(round(time() * 1000))

print(t0)

# Looping Simulation
for i in range(0, len(t)):
    # print("data ke-",i," = ",x1[i])
    # print("waktu ke-",i," = ",t[i])
    if (abs(x1[i]) < toleransi_sudut and rt == t_end):
        rt = t[i]
        st = t[i]
    if (t[i] > rt and abs(x1[i]) > toleransi_sudut):
        OvS = x1[i]
        if (abs(OvS) < abs(OvBef) ):
            peakOv[OvKe] = OvBef
        else:
            OvBef = OvS
    if (t[i] > rt and peakOv[OvKe] > 0):
        OvKe = OvKe +1
        OvS = 0
        OvBef = 0
    
    # menghitung cost function
    t1 = time()*1000
    dt = t1 - t0
#    print("t0 = ",t0/1000)
#    print("t1 = ",t1/1000)
#    print("dt = ",dt/1000)
    t0 = t1
    xCF = np.array([[x1[i]],[x2[i]],[x3[i]],[x4[i]],[x5[i]],[x6[i]]])
    u = -K@xCF
#    print("u = ", u)
    J = J + ((np.transpose(xCF)@Q@xCF) + (np.transpose(u)@R@u))*dt/1000
#    print("J ke-",i," = ",J)

# cost funtion
print("Rise Time = ",rt)
print("Overshoot =\n",peakOv)
# print("Cost Function = ",J)

# rt = 0;

# tol_awal_roll = 5/100*90;
# tol_awal_pitch = 5/100*90;
# tol_awal_yaw = 5/100*180;

# u=zeros(size(t));

# nilai input proses
# u1 = [1 0 0 ]*(-K*x');
# u2 = [0 1 0 ]*(-K*x');
# u3 = [0 0 1 ]*(-K*x');

# Simulasi Anti Roll
# x3=[1 0 0 0 0 0 ]*x'; %gangguan roll

# figure(1)
# inf1 = stepinfo(x3)
# degnya = 0;
# for v = 1:1:101
#     degnya = x3(1,v);
#     degnya = rad2deg(degnya);
#       if degnya <= 4.5;
#             rt = (v-1)/10; %rt = rise time
#             break;
#       end
# end
# w = -35:0.1:35;
# plot((rt*ones(size(w))),w)
# hold on;


# plot(t,(0*tol_awal_roll*ones(size(t))))
# hold on;

# plot(t,(-1*tol_awal_roll*ones(size(t))))
# hold on;

# plot(t,(1*tol_awal_roll*ones(size(t))))
# hold on;


# plot (t,rad2deg(x3))
# hold off;

# legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Roll')

# ylim ([-20, 30])
# grid
# title('Respons Sudut Penstabil-Roll')
# xlabel('Waktu')
# ylabel('Sudut Wahana')

# %Simulasi Anti Pitch
# x4=[0 0 1 0 0 0]*x'; %gangguan pitch
# inf2 = stepinfo(x4)
# figure(2)

# degnya = 0;
# for v = 1:1:101
#     degnya = x4(1,v);
#     degnya = rad2deg(degnya);
#       if degnya <= 4.5;
#             rt = (v-1)/10;
#             break;
#       end
# end
# w = -35:0.1:35;
# plot((rt*ones(size(w))),w)
# hold on;


# plot(t,(0*tol_awal_pitch*ones(size(t))))
# hold on;

# plot(t,(-1*tol_awal_pitch*ones(size(t))))
# hold on;

# plot(t,(1*tol_awal_pitch*ones(size(t))))
# hold on;


# plot (t,rad2deg(x4))
# hold off;

# legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Pitch')
# ylim ([-20, 30])
# grid
# title('Respons Sudut Penstabil-Pitch')
# xlabel('Waktu')
# ylabel('Sudut Wahana')

# %Simulasi Anti Yaw
# x5=[ 0 0 0 0 1 0]*x'; %gangguan yaw

# figure(3)
# inf3 = stepinfo(x5,'RiseTimeLimits',[0.05,0.95])
# degnya = 0;
# for v = 1:1:101
#     degnya = x5(1,v);
#     degnya = rad2deg(degnya);
#       if degnya <= 4.5;
#             rt = (v-1)/10;
#             break;
#       end
# end
# w = -35:0.1:35;
# plot((rt*ones(size(w))),w)
# hold on;


# plot(t,(0*tol_awal_yaw*ones(size(t))))
# hold on;

# plot(t,(-1*tol_awal_yaw*ones(size(t))))
# hold on;

# plot(t,(1*tol_awal_yaw*ones(size(t))))
# hold on;


# plot (t,rad2deg(x5))
# hold off;

# legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Yaw')
# ylim ([-20, 30])
# grid
# title('Respons Sudut Penstabil-Yaw')
# xlabel('Waktu')
# ylabel('Sudut Wahana')
