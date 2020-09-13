function [J grad]=Brain(Theta,alpha_len,beta_len,gamma_len,alpha_hidden_layer1_node,alpha_hidden_layer2_node,alpha_hidden_layer3_node,alpha_hidden_layer4_node,alpha_lambda,beta_hidden_layer1_node,beta_hidden_layer2_node,beta_lambda,gamma_hidden_layer1_node,gamma_hidden_layer2_node,gamma_lambda,XH,XL,yH,yL,alpha_input_layer_node, beta_input_layer_node,gamma_input_layer_node,alpha_output_layer_node, beta_output_layer_node,gamma_output_layer_node)

ThetaL = reshape(Theta(1:alpha_len),alpha_len,1);
ThetaLin = reshape(Theta(1+alpha_len:alpha_len+beta_len),beta_len,1);
ThetaNL = reshape(Theta(1+alpha_len+beta_len:alpha_len+beta_len+gamma_len),gamma_len,1);

alpha1=Theta(end-1);
alpha2=Theta(end);

Theta1L = reshape(ThetaL(1:alpha_hidden_layer1_node*(alpha_input_layer_node+1)),alpha_input_layer_node+1,alpha_hidden_layer1_node);
Theta2L = reshape(ThetaL(1+alpha_hidden_layer1_node*(alpha_input_layer_node+1):alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node),alpha_hidden_layer1_node+1,alpha_hidden_layer2_node);
Theta3L = reshape(ThetaL(alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+1:alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node),(alpha_hidden_layer2_node+1),alpha_hidden_layer3_node);
Theta4L = reshape(ThetaL(alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node+1:alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node+(alpha_hidden_layer3_node+1)*alpha_hidden_layer4_node),(alpha_hidden_layer3_node+1),alpha_hidden_layer4_node);
Theta5L = reshape(ThetaL(alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node+(alpha_hidden_layer3_node+1)*alpha_hidden_layer4_node+1:alpha_hidden_layer1_node*(alpha_input_layer_node+1)+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node+(alpha_hidden_layer3_node+1)*alpha_hidden_layer4_node+(alpha_hidden_layer4_node+1)*alpha_output_layer_node),(alpha_hidden_layer4_node+1),alpha_output_layer_node);

Theta1Lin = reshape(ThetaLin(1:beta_hidden_layer1_node * (beta_input_layer_node + 1)),(beta_input_layer_node + 1),beta_hidden_layer1_node);
Theta2Lin = reshape(ThetaLin((1 + (beta_hidden_layer1_node * (beta_input_layer_node + 1))):((beta_hidden_layer1_node * (beta_input_layer_node + 1)))+(beta_hidden_layer1_node+1)*(beta_hidden_layer2_node)),beta_hidden_layer1_node+1, beta_hidden_layer2_node);
Theta3Lin = reshape(ThetaLin(((beta_hidden_layer1_node * (beta_input_layer_node + 1)))+(beta_hidden_layer1_node+1)*(beta_hidden_layer2_node)+1:end),(beta_hidden_layer2_node + 1),beta_output_layer_node);

Theta1NL = reshape(ThetaNL(1:gamma_hidden_layer1_node * (gamma_input_layer_node + 1)),(gamma_input_layer_node + 1),gamma_hidden_layer1_node);
Theta2NL = reshape(ThetaNL((1 + (gamma_hidden_layer1_node * (gamma_input_layer_node + 1))):((gamma_hidden_layer1_node * (gamma_input_layer_node + 1)))+(gamma_hidden_layer1_node+1)*(gamma_hidden_layer2_node)),gamma_hidden_layer1_node+1, gamma_hidden_layer2_node);
Theta3NL = reshape(ThetaNL(((gamma_hidden_layer1_node * (gamma_input_layer_node + 1)))+(gamma_hidden_layer1_node+1)*(gamma_hidden_layer2_node)+1:end),(gamma_hidden_layer2_node + 1),gamma_output_layer_node);



[JL gradL]=Neural(XL,yL,alpha_hidden_layer1_node,alpha_hidden_layer2_node,alpha_hidden_layer3_node,alpha_hidden_layer4_node,alpha_lambda,ThetaL);
yL_trained=trained(XH, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_beta=[XH; yL_trained];
[JLin gradLin grad_alpha1]=Neural_L(X_beta,yH,beta_hidden_layer1_node,beta_hidden_layer2_node,beta_lambda,ThetaLin,alpha1);
yL_trained=trained(XH, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_gamma=[XH; yL_trained];
[JNL gradNL grad_alpha2]=Neural_NL(X_gamma,yH,gamma_hidden_layer1_node,gamma_hidden_layer2_node,gamma_lambda,ThetaNL,alpha2);
h_Lin=trained_L(X_beta,Theta1Lin,Theta2Lin,Theta3Lin);
h_NL=trained_NL(X_gamma,Theta1NL,Theta2NL,Theta3NL);
h=h_Lin+h_NL;
Jalpha=sum(sum((yH-h).^2))/(length(h(1,:)));

if alpha1>1||alpha2<0
alpha1=1;
alpha2=0;
elseif alpha2>1||alpha1<0
alpha1=0;
alpha2=1;
else
end


J=JL+JLin+JNL+Jalpha;
grad = [gradL(:) ; gradLin(:) ; gradNL(:) ; grad_alpha1 ; grad_alpha2];
end