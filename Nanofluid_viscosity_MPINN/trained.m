function h=trained(x, Theta1, Theta2, Theta3,Theta4,Theta5)

m=length(x(1,:));
X1=x;
a1=X1;
a1=[ones(1,m); a1];
z2=Theta1'*a1;
a2=sigmoid(z2);
a2=[ones(1,m); a2];
z3=Theta2'*a2;
a3=sigmoid(z3);
a3=[ones(1,m); a3];
z4=Theta3'*a3;
a4=sigmoid(z4);
a4=[ones(1,m); a4];
z5=Theta4'*a4;
a5=sigmoid(z5);
a5=[ones(1,m); a5];
z6=Theta5'*a5;
a6=linear1(z6);
%a4=sigmoid(z4);
h=a6;

end