function [J grad]=Neural(X,y,hidden_layer1_node,hidden_layer2_node,hidden_layer3_node,hidden_layer4_node,lambda,Theta)

input_layer_node=length(X(:,1));
output_layer_node=length(y(:,1));

i=input_layer_node;
o=output_layer_node;
h1=hidden_layer1_node;
h2=hidden_layer2_node;
h3=hidden_layer3_node;
h4=hidden_layer4_node;

Theta1 = reshape(Theta(1:h1*(i+1)),i+1,h1);
Theta2 = reshape(Theta(1+h1*(i+1):h1*(i+1)+(h1+1)*h2),h1+1,h2);
Theta3 = reshape(Theta(h1*(i+1)+(h1+1)*h2+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3),(h2+1),h3);
Theta4 = reshape(Theta(h1*(i+1)+(h1+1)*h2+(h2+1)*h3+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4),(h3+1),h4);
Theta5 = reshape(Theta(h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4+(h4+1)*o),(h4+1),o);

m=length(X(1,:));
X1=[ones(1,m); X];

J=0;
a1=X1;
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
c1=J;
J=sum(sum((y-h).^2))/(m);

t1 = Theta1(2:size(Theta1,1),:);
t2 = Theta2(2:size(Theta2,1),:);
t3 = Theta3(2:size(Theta3,1),:);
t4 = Theta4(2:size(Theta4,1),:);
t5 = Theta5(2:size(Theta5,1),:);

Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 )) + sum( sum ( t3.^2))+sum( sum ( t4.^2))+sum( sum ( t5.^2))) / (2*m);
J=J+Reg;

delta6=h-y;
z5=[ones(1,m); z5];
delta5=(Theta5*delta6).*linearGradient(z5);
delta5 = delta5(2:end,:);
z4=[ones(1,m); z4];
delta4=(Theta4*delta5).*sigmoidGradient(z4);
delta4 = delta4(2:end,:);
z3=[ones(1,m); z3];
delta3=(Theta3*delta4).*sigmoidGradient(z3);
%delta3=(Theta3*delta4).*sigmoidGradient(z3);
delta3 = delta3(2:end,:);
z2=[ones(1,m); z2];
delta2=(Theta2*delta3).*sigmoidGradient(z2);
delta2 = delta2(2:end,:);


Theta1_grad=a1*delta2';
Theta2_grad=a2*delta3';
Theta3_grad=a3*delta4';
Theta4_grad=a4*delta5';
Theta5_grad=a5*delta6';

Theta3_grad = (1/m) * Theta3_grad;
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;
Theta4_grad = (1/m) * Theta4_grad;
Theta5_grad = (1/m) * Theta5_grad;

Theta1_grad(2:end,:) = Theta1_grad(2:end,:) + ((lambda/m) * Theta1(2:end,:));
Theta2_grad(2:end,:) = Theta2_grad(2:end,:) + ((lambda/m) * Theta2(2:end,:));
Theta3_grad(2:end,:) = Theta3_grad(2:end,:) + ((lambda/m) * Theta3(2:end,:));
Theta4_grad(2:end,:) = Theta4_grad(2:end,:) + ((lambda/m) * Theta4(2:end,:));
Theta5_grad(2:end,:) = Theta5_grad(2:end,:) + ((lambda/m) * Theta5(2:end,:));

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:) ; Theta4_grad(:) ; Theta5_grad(:)];
end
