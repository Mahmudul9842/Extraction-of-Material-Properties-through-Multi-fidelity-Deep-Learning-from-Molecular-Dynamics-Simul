function h=trained_NL(x, Theta1, Theta2, Theta3)

m=length(x(1,:));
X1=[ones(1,m); x];
a1=X1;
z2=Theta1'*a1;
a2=sigmoid(z2);
a2=[ones(1,m); a2];
z3=Theta2'*a2;
a3=sigmoid(z3);
a3=[ones(1,m); a3];
z4=Theta3'*a3;
a4=sigmoid(z4);
h=a4;

end