clear;
load('C:/15 Batch Thesis/Ridzy/brain/workspace_v.mat')
i=1
for x=0:0.01:1
    for y=0:0.01:1
        mm(i)=x;
        nn(i)=y;
        i=i+1;
    end
end
T_phi=[mm; nn];


yL_trained=trained(T_phi, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_beta=[T_phi; yL_trained];
X_gamma=[T_phi; yL_trained];
h_Lin=trained_L(X_beta,Theta1Lin,Theta2Lin,Theta3Lin);
h_NL=trained_NL(X_gamma,Theta1NL,Theta2NL,Theta3NL);
h=(h_Lin+h_NL);
max_T=102;
min_T=86;
max_phi=4;
min_phi=0;

mm=mm*(max_T-min_T)+min_T;
nn=nn*(max_phi-min_phi)+min_phi;
max_H= 3.21746406e-04;
min_H= 2.30803047e-04;
h=h*(max_H-min_H)+min_H;
final=[mm; nn; h]

fid = fopen('data.txt','w');
g=length(h);
for i=1:1:g
fprintf(fid,'%f %f %.12f\n',mm(i),nn(i),h(i));
end
fclose(fid)