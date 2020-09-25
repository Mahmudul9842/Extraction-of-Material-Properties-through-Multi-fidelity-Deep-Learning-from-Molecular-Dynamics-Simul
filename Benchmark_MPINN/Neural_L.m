function [J grad grad_alpha1]=Neural_L(X,y,hidden_layer1_node,hidden_layer2_node,lambda,Theta,alpha1)

input_layer_node=length(X(:,1));
output_layer_node=length(y(:,1));
Theta1 = reshape(Theta(1:hidden_layer1_node * (input_layer_node + 1)),(input_layer_node + 1),hidden_layer1_node);
Theta2 = reshape(Theta((1 + (hidden_layer1_node * (input_layer_node + 1))):((hidden_layer1_node * (input_layer_node + 1)))+(hidden_layer1_node+1)*(hidden_layer2_node)),hidden_layer1_node+1, hidden_layer2_node);
Theta3 = reshape(Theta(((hidden_layer1_node * (input_layer_node + 1)))+(hidden_layer1_node+1)*(hidden_layer2_node)+1:end),(hidden_layer2_node + 1),output_layer_node);

m=length(X(1,:));
X1=[ones(1,m); X];

J=0;
a1=X1;
z2=Theta1'*a1;
a2=z2;
a2=[ones(1,m); a2];
z3=Theta2'*a2;
a3=z3;
a3=[ones(1,m); a3];
z4=Theta3'*a3;
a4=z4;
h=a4;
c1=J;
J=sum(sum((y*alpha1-h).^2))/(m);
grad_alpha1=sum(sum((h-y*alpha1).*y))/m;
t1 = Theta1(2:size(Theta1,1),:);
t2 = Theta2(2:size(Theta2,1),:);
t3 = Theta3(2:size(Theta3,1),:);
Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 )) + sum( sum ( t3.^2))) / (2*m);
J=J+Reg;

delta4=h-y*alpha1;
%delta4 = delta4(:,2:end);
z3=[ones(1,m); z3];
delta3=Theta3*delta4;
delta3 = delta3(2:end,:);
z2=[ones(1,m); z2];
delta2=(Theta2*delta3);
delta2 = delta2(2:end,:);


Theta1_grad=a1*delta2';
Theta2_grad=a2*delta3';
Theta3_grad=a3*delta4';

Theta3_grad = (1/m) * Theta3_grad;
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

Theta1_grad(2:end,:) = Theta1_grad(2:end,:) + ((lambda/m) * Theta1(2:end,:));
Theta2_grad(2:end,:) = Theta2_grad(2:end,:) + ((lambda/m) * Theta2(2:end,:));
Theta3_grad(2:end,:) = Theta3_grad(2:end,:) + ((lambda/m) * Theta3(2:end,:));


grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
end
