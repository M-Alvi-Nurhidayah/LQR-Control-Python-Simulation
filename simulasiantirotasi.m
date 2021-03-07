Ixx = 0.103557335;
Iyy = 0.129538628;
Izz = 0.219640939;
		
m = 1.78;
L1  = 1.435;%panjang sayap yang sejajar dengan horzontal stab (dari CG).
L2  = 0.66;% Panjang badan (dari CG) sampai ujung ekor pesawat

%nilai state yang menjadi variabel dalam matrix A dianggap kecil
r = 0.000001;
p = 0.000001;
q = 0.000001;
      % 1   2   3   4   5   6   7   8   9   10
A =  [  0   1   0   0   0   0;          %Roll
        0   0   0 (Iyy-Izz)*r/Ixx  0 0; %Kec.Sudut Roll
        0   0   0   1   0   0;          %Pitch
        0 (Izz-Ixx)*r/Iyy  0 0 0 0;     %Kec. Sudut Pitch
        0   0   0   0   0   1;          %Yaw
        0 (Ixx-Iyy)*q/Izz  0 0 0 0];    %Kec. Sudut yaw

  %     1   2   3   4   5  
B = [   0   0   0
        1/Ixx 0 0;
        0   0   0;
        0 1/Iyy 0;
        0   0   0;
        0   0 1/Izz ];
    
 %	 1  2   3   4   5   6   7   8   9   10      
C = [   1   0   0   0   0   0; 
        0   0   1   0   0   0;
        0   0   0   0   1   0];
 
  %     1   2   3   4   5 
 D = zeros(3,3);
     
 %nilai untuk ditelaa agar mendapatkan respon sistem yang sesuai
    %   1    2   3   4    5   6   7   8   9   10      
Q = [   27    0    0   0   0   0; %Roll
        0       0.7  0   0   0   0; %gyro roll
        0       0   25  0   0   0; %pitch
        0       0    0  0.4 0   0; %gyro pitch
        0       0    0   0   43  0; %yaw
        0       0    0   0   0   0.85];%gyro yaw
%%
%Nilai fullstate feedback gain

R = eye(3); %mengikuti jumlah u
ws = 0.0025 % waktu sampel
[K,SS,ee] = lqrd(A,B,Q,R,ws) %lqr discrete
%https://www.mathworks.com/help/control/ref/lqrd.html

Ac = [(A-B*K)];
Bc = eye(6);
Cc = eye(6);
Dc = 0;

sys_cl = ss(Ac,Bc,Cc,Dc);
sys_ol = ss(A,B,C,D);

info = stepinfo(sys_cl)

poles=eig(A) %kalo polesnya positif sulit dikendaliin

control=ctrb(sys_cl)
controllability=rank(control) %matriks keterkendalian besarnya harus sama seperti matriks A

roll = deg2rad(30); %besar sudut gangguan awal
pitch = deg2rad(30);
yaw = deg2rad(30);


x0=[roll;0;pitch;0;yaw;0]; %gangguan awal sistem
t=0:0.1:3;
x=initial(sys_cl,x0,t);

rt = 0;

tol_awal_roll = 5/100*90;
tol_awal_pitch = 5/100*90;
tol_awal_yaw = 5/100*180;

u=zeros(size(t));

%nilai input proses
u1 = [1 0 0 ]*(-K*x');
u2 = [0 1 0 ]*(-K*x');
u3 = [0 0 1 ]*(-K*x');



%%

%Simulasi Anti Roll
x3=[1 0 0 0 0 0 ]*x'; %gangguan roll

figure(1)
inf1 = stepinfo(x3)
degnya = 0;
for v = 1:1:101
    degnya = x3(1,v);
    degnya = rad2deg(degnya);
      if degnya <= 4.5;
            rt = (v-1)/10; %rt = rise time
            break;
      end
end
w = -35:0.1:35;
plot((rt*ones(size(w))),w)
hold on;


plot(t,(0*tol_awal_roll*ones(size(t))))
hold on;

plot(t,(-1*tol_awal_roll*ones(size(t))))
hold on;

plot(t,(1*tol_awal_roll*ones(size(t))))
hold on;


plot (t,rad2deg(x3))
hold off;

legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Roll')

ylim ([-20, 30])
grid
title('Respons Sudut Penstabil-Roll')
xlabel('Waktu')
ylabel('Sudut Wahana')

%%
%Simulasi Anti Pitch
x4=[0 0 1 0 0 0]*x'; %gangguan pitch
inf2 = stepinfo(x4)
figure(2)

degnya = 0;
for v = 1:1:101
    degnya = x4(1,v);
    degnya = rad2deg(degnya);
      if degnya <= 4.5;
            rt = (v-1)/10;
            break;
      end
end
w = -35:0.1:35;
plot((rt*ones(size(w))),w)
hold on;


plot(t,(0*tol_awal_pitch*ones(size(t))))
hold on;

plot(t,(-1*tol_awal_pitch*ones(size(t))))
hold on;

plot(t,(1*tol_awal_pitch*ones(size(t))))
hold on;


plot (t,rad2deg(x4))
hold off;

legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Pitch')
ylim ([-20, 30])
grid
title('Respons Sudut Penstabil-Pitch')
xlabel('Waktu')
ylabel('Sudut Wahana')

%%
%Simulasi Anti Yaw
x5=[ 0 0 0 0 1 0]*x'; %gangguan yaw

figure(3)
inf3 = stepinfo(x5,'RiseTimeLimits',[0.05,0.95])
degnya = 0;
for v = 1:1:101
    degnya = x5(1,v);
    degnya = rad2deg(degnya);
      if degnya <= 4.5;
            rt = (v-1)/10;
            break;
      end
end
w = -35:0.1:35;
plot((rt*ones(size(w))),w)
hold on;


plot(t,(0*tol_awal_yaw*ones(size(t))))
hold on;

plot(t,(-1*tol_awal_yaw*ones(size(t))))
hold on;

plot(t,(1*tol_awal_yaw*ones(size(t))))
hold on;


plot (t,rad2deg(x5))
hold off;

legend('Rise Time','Setpoint','Toleransi Bawah','Toleransi Atas','Yaw')
ylim ([-20, 30])
grid
title('Respons Sudut Penstabil-Yaw')
xlabel('Waktu')
ylabel('Sudut Wahana')
