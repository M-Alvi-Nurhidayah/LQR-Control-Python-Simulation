# SIMULASI SISTEM KENDALI LQR 
# Penelitian Skripsi L1 Controller Navigation
# M Alvi Nurhidayah

import numpy as np
import control as ctl
import control.matlab as matlab
import matplotlib.pyplot as plt
from time import time

# =====================> Fixedwing Configuration
M_body = 1.276
p_body = 0.1
l_body = 0.94
t_body = 0.165
x_body = 1
y_body = 1
z_body = 1
M_sayap = 0.38
p_sayap = 1.435
l_sayap = 0.22
t_sayap = 0.025
x_sayap = 1
y_sayap = 1
z_sayap = 1
M_Hstab = 0.079
p_Hstab = 0.001
l_Hstab = 0.185
t_Hstab = 0.015
x_Hstab = 1
y_Hstab = 1
z_Hstab = 1
M_Vstab = 0.01
p_Vstab = 0.005
l_Vstab = 0.332
t_Vstab = 0.227
x_Vstab = 1
y_Vstab = 1
z_Vstab = 1
M_motor = 0.5
r_motor = 0.002
t_motor = 0.002
x_motor = 1
y_motor = 1
z_motor = 1
M_prop = 0.015
r_prop = 0.11 
t_prop = 0.015
x_prop = 1
y_prop = 1
z_prop = 1
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
print("inersia Z : {:.3f}".format(Izz))

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
print("nilai q : {:.3f}".format(q))

# State Space Modelling
A = np.array([[0,        1,         0,         0,         0,       0],   # Roll
              [0,        0,         0, (Iyy-Izz)*r/Ixx,   0,       0],   # Kec. Sudut Roll
              [0,        0,         0,         1,         0,       0],   # Pitch
              [0, (Izz-Ixx)*r/Iyy,  0,         0,         0,       0],   # Kec. Sudut Pitch
              [0,        0,         0,         0,         0,       1],   # Yaw
              [0, (Ixx-Iyy)*q/Izz,  0,         0,         0,       0]])  # Kec. Sudut Yaw
eigen_value = np.linalg.eigvals(A)
print("A = \n{}".format(np.around(A, 4)))
print("EigenValue = \n{}".format(eigen_value))

B = np.array([[0,       0,       0],
              [1/Ixx,   0,       0],
              [0,       0,       0],
              [0,     1/Iyy,     0],
              [0,       0,       0],
              [0,       0,   1/Izz]])
print("B = \n{}".format(np.around(B, 4)))

C = np.array([[ 1,   0,   0,   0,   0,   0,],
              [ 0,   0,   1,   0,   0,   0,],
              [ 0,   0,   0,   0,   1,   0,]])
print("C = \n{}".format(np.around(C, 4)))

D = np.zeros((3,3))

# =====================> Sistem Kendali LQR     
# Nilai untuk ditelaa agar mendapatkan respon sistem yang sesuai
Q = np.array([[ 1,   0,   0,   0,   0,   0],     #Roll
              [ 0,   1,   0,   0,   0,   0],     #gyro roll
              [ 0,   0,   1,   0,   0,   0],     #pitch
              [ 0,   0,   0,   1,   0,   0],     #gyro pitch
              [ 0,   0,   0,   0,   1,   0],     #yaw
              [ 0,   0,   0,   0,   0,   1]])    #gyro yaw
print("Q = \n{}".format(np.around(Q, 4)))

# Matriks R mengikuti jumlah u
R = np.array([[ 1,   0,   0],
              [ 0,   1,   0],
              [ 0,   0,   1]])
print("R = \n{}".format(np.around(A, 4)))

K, S, E = ctl.lqr(A, B, Q, R)
print("K = \n", np.round(K, 4))
print("Eigen Value = \n{}".format(E))

# Simulation
t_start = 0
t_end = 10
dt = 0.01
t = np.arange(t_start,t_end+dt,dt)     # deret waktu 0 - 5 dengan kenaikan dt (0,01)

# ws = 0.0025 % waktu sampel
# [K,SS,ee] = lqrd(A,B,Q,R,ws) %lqr discrete
# # https://www.mathworks.com/help/control/ref/lqrd.html

# Ac = [(A-B*K)];
# Bc = eye(6);
# Cc = eye(6);
# Dc = 0;

# sys_cl = ss(Ac,Bc,Cc,Dc);
# sys_ol = ss(A,B,C,D);

# info = stepinfo(sys_cl)

# poles=eig(A) %kalo polesnya positif sulit dikendaliin

# control=ctrb(sys_cl)
# controllability=rank(control) %matriks keterkendalian besarnya harus sama seperti matriks A

# roll = deg2rad(30); %besar sudut gangguan awal
# pitch = deg2rad(30);
# yaw = deg2rad(30);


# x0=[roll;0;pitch;0;yaw;0]; %gangguan awal sistem
# t=0:0.1:3;
# x=initial(sys_cl,x0,t);

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
