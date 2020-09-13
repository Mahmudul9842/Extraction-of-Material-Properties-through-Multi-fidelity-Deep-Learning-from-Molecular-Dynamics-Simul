for ridzy=1:10
for mm=10:10:90

XL=load('T_rho_low_fidelity.txt');
XL=XL';
yL=load('P_low_fidelity.txt'); 
yL=yL';
filename1=sprintf('T_rho_high_fidelity_%d.txt',mm);
%filename1=sprintf('T_rho_high_fidelity.txt');
XH=load(filename1); %load input data
XH=XH';
XH1=load('T_rho_high_fidelity.txt');
XH1=XH1';
filename2=sprintf('P_high_fidelity_%d.txt',mm);
%filename2=sprintf('P_high_fidelity.txt');
yH=load(filename2); %load output data
yH=yH';
yH1=load('P_high_fidelity.txt');
yH1=yH1';
%y=[yL yH];
m=length(XH1(1,:));
%idx = randperm(m);
%data_per=mm/100;
%new_m=floor(m*data_per);
%XH1=XH;
%XH(:,idx) = XH1(:,:);
%XH=XH(:,1:new_m);
%yH1 = yH ;
%yH(:,idx) = yH1(:,:);
%yH=yH(:,1:new_m);

%normalize%
max_T=max(XL(1,:));
min_T=min(XL(1,:));
max_rho=max(XL(2,:));
min_rho=min(XL(2,:));

XH(1,:)=(XH(1,:)-min_T)/(max_T-min_T);
XH(2,:)=(XH(2,:)-min_rho)/(max_rho-min_rho);
XL(1,:)=(XL(1,:)-min_T)/(max_T-min_T);
XL(2,:)=(XL(2,:)-min_rho)/(max_rho-min_rho);
XH1(1,:)=(XH1(1,:)-min_T)/(max_T-min_T);
XH1(2,:)=(XH1(2,:)-min_rho)/(max_rho-min_rho);


%maxH=150;
%minH=0;
maxH=max(yH);
minH=min(yH);

yH=(yH-minH)/(maxH-minH);
yH1=(yH1-minH)/(maxH-minH);

%maxL=150;
%minL=0;
maxL=max(yL);
minL=min(yL);

yL=(yL-minL)/(maxL-minL);

alpha1=0.1;
alpha2=0.9;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ALPHA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha_input_layer_node=length(XL(:,1));
alpha_output_layer_node=length(yL(:,1));
alpha_hidden_layer1_node=20;
alpha_hidden_layer2_node=20;
alpha_hidden_layer3_node=20;
alpha_hidden_layer4_node=20;

Theta1L = randInitializeWeights(alpha_hidden_layer1_node-1,alpha_input_layer_node+1);
Theta2L = randInitializeWeights(alpha_hidden_layer2_node-1, alpha_hidden_layer1_node+1);
Theta3L = randInitializeWeights(alpha_hidden_layer3_node-1, alpha_hidden_layer2_node+1);
Theta4L = randInitializeWeights(alpha_hidden_layer4_node-1, alpha_hidden_layer3_node+1);
Theta5L = randInitializeWeights(alpha_output_layer_node-1,alpha_hidden_layer4_node+1);

alpha_lambda=0;

ThetaL = [Theta1L(:) ; Theta2L(:) ; Theta3L(:) ; Theta4L(:) ; Theta5L(:)];


%[JL gradL]=Neural(XL,yL,alpha_hidden_layer1_node,alpha_hidden_layer2_node,alpha_lambda,ThetaL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BETA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yL_trained=trained(XH, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_beta=[XH; yL_trained];

beta_input_layer_node=length(X_beta(:,1));
beta_output_layer_node=length(yH(:,1));
beta_hidden_layer1_node=1;
beta_hidden_layer2_node=1;

Theta1Lin = randInitializeWeights(beta_hidden_layer1_node-1,beta_input_layer_node+1);
Theta2Lin = randInitializeWeights(beta_hidden_layer2_node-1, beta_hidden_layer1_node+1);
Theta3Lin = randInitializeWeights(beta_output_layer_node-1,beta_hidden_layer2_node+1);

beta_lambda=0;

ThetaLin = [Theta1Lin(:) ; Theta2Lin(:) ; Theta3Lin(:)];

%[h_Lin gradLin Reg_Lin]=Neuralg_L(X_beta,yH,beta_hidden_layer1_node,beta_hidden_layer2_node,beta_lambda,ThetaLin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAMMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

yL_trained=trained(XH, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_gamma=[XH; yL_trained];

gamma_input_layer_node=length(X_gamma(:,1));
gamma_output_layer_node=length(yH(:,1));
gamma_hidden_layer1_node=20;
gamma_hidden_layer2_node=20;


Theta1NL = randInitializeWeights(gamma_hidden_layer1_node-1,gamma_input_layer_node+1);
Theta2NL = randInitializeWeights(gamma_hidden_layer2_node-1, gamma_hidden_layer1_node+1);
Theta3NL = randInitializeWeights(gamma_output_layer_node-1,gamma_hidden_layer2_node+1);

gamma_lambda=0.001;

ThetaNL = [Theta1NL(:) ; Theta2NL(:) ; Theta3NL(:)];

gamma_len=(gamma_input_layer_node+1)*gamma_hidden_layer1_node+(gamma_hidden_layer1_node+1)*gamma_hidden_layer2_node+(gamma_hidden_layer2_node+1)*gamma_output_layer_node;
beta_len=(beta_input_layer_node+1)*beta_hidden_layer1_node+(beta_hidden_layer1_node+1)*beta_hidden_layer2_node+(beta_hidden_layer2_node+1)*beta_output_layer_node;
alpha_len=(alpha_input_layer_node+1)*alpha_hidden_layer1_node+(alpha_hidden_layer1_node+1)*alpha_hidden_layer2_node+(alpha_hidden_layer2_node+1)*alpha_hidden_layer3_node+(alpha_hidden_layer3_node+1)*alpha_hidden_layer4_node+(alpha_hidden_layer4_node+1)*alpha_output_layer_node;

%[h_NL gradNL Reg_NL]=Neuralg_NL(X_gamma,yH,gamma_hidden_layer1_node,gamma_hidden_layer2_node,gamma_lambda,ThetaNL);
Theta = [ThetaL(:) ; ThetaLin(:) ; ThetaNL(:) ; alpha1 ; alpha2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[J grad]=Brain(Theta,alpha_len,beta_len,gamma_len,alpha_hidden_layer1_node,alpha_hidden_layer2_node,alpha_hidden_layer3_node,alpha_hidden_layer4_node,alpha_lambda,beta_hidden_layer1_node,beta_hidden_layer2_node,beta_lambda,gamma_hidden_layer1_node,gamma_hidden_layer2_node,gamma_lambda,XH,XL,yH,yL,alpha_input_layer_node, beta_input_layer_node,gamma_input_layer_node,alpha_output_layer_node, beta_output_layer_node,gamma_output_layer_node);
%ll=ceil(0.3125*mm+1.875);
for i=1:2
options = optimset('MaxIter', 10000 , 'Display', 'iter', 'MaxFunEvals', 1000000,'TolFun',1e-50,'TolX',1e-50);
costFunction = @(p) Brain(p,alpha_len,beta_len,gamma_len,alpha_hidden_layer1_node,alpha_hidden_layer2_node,alpha_hidden_layer3_node,alpha_hidden_layer4_node,alpha_lambda,beta_hidden_layer1_node,beta_hidden_layer2_node,beta_lambda,gamma_hidden_layer1_node,gamma_hidden_layer2_node,gamma_lambda,XH,XL,yH,yL,alpha_input_layer_node, beta_input_layer_node,gamma_input_layer_node,alpha_output_layer_node, beta_output_layer_node,gamma_output_layer_node);
[Theta, cost] = fminlbfgs(costFunction, Theta, options);
%[Theta, cost, exitflag, output] = fmin_adam(costFunction, Theta, 0.01);
%[Theta, cost] = fmin_adam(costFunction, Theta, options);
if cost<1e-6
    break;
else
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ThetaL = reshape(Theta(1:alpha_len),alpha_len,1);
ThetaLin = reshape(Theta(1+alpha_len:alpha_len+beta_len),beta_len,1);
ThetaNL = reshape(Theta(1+alpha_len+beta_len:alpha_len+beta_len+gamma_len),gamma_len,1);
alpha1=Theta(end-1);
alpha2=Theta(end);
if alpha1>1||alpha2<0
alpha1=1;
alpha2=0;
elseif alpha2>1||alpha1<0
alpha1=0;
alpha2=1;
else
end

i=alpha_input_layer_node;
o=alpha_output_layer_node;
h1=alpha_hidden_layer1_node;
h2=alpha_hidden_layer2_node;
h3=alpha_hidden_layer3_node;
h4=alpha_hidden_layer4_node;

Theta1L = reshape(ThetaL(1:h1*(i+1)),i+1,h1);
Theta2L = reshape(ThetaL(1+h1*(i+1):h1*(i+1)+(h1+1)*h2),h1+1,h2);
Theta3L = reshape(ThetaL(h1*(i+1)+(h1+1)*h2+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3),(h2+1),h3);
Theta4L = reshape(ThetaL(h1*(i+1)+(h1+1)*h2+(h2+1)*h3+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4),(h3+1),h4);
Theta5L = reshape(ThetaL(h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4+1:h1*(i+1)+(h1+1)*h2+(h2+1)*h3+(h3+1)*h4+(h4+1)*o),(h4+1),o);


Theta1Lin = reshape(ThetaLin(1:beta_hidden_layer1_node * (beta_input_layer_node + 1)),(beta_input_layer_node + 1),beta_hidden_layer1_node);
Theta2Lin = reshape(ThetaLin((1 + (beta_hidden_layer1_node * (beta_input_layer_node + 1))):((beta_hidden_layer1_node * (beta_input_layer_node + 1)))+(beta_hidden_layer1_node+1)*(beta_hidden_layer2_node)),beta_hidden_layer1_node+1, beta_hidden_layer2_node);
Theta3Lin = reshape(ThetaLin(((beta_hidden_layer1_node * (beta_input_layer_node + 1)))+(beta_hidden_layer1_node+1)*(beta_hidden_layer2_node)+1:end),(beta_hidden_layer2_node + 1),beta_output_layer_node);

Theta1NL = reshape(ThetaNL(1:gamma_hidden_layer1_node * (gamma_input_layer_node + 1)),(gamma_input_layer_node + 1),gamma_hidden_layer1_node);
Theta2NL = reshape(ThetaNL((1 + (gamma_hidden_layer1_node * (gamma_input_layer_node + 1))):((gamma_hidden_layer1_node * (gamma_input_layer_node + 1)))+(gamma_hidden_layer1_node+1)*(gamma_hidden_layer2_node)),gamma_hidden_layer1_node+1, gamma_hidden_layer2_node);
Theta3NL = reshape(ThetaNL(((gamma_hidden_layer1_node * (gamma_input_layer_node + 1)))+(gamma_hidden_layer1_node+1)*(gamma_hidden_layer2_node)+1:end),(gamma_hidden_layer2_node + 1),gamma_output_layer_node);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%T=[];
%rho=[];
%j=1;
%for i=50:1:1000
 %  for z=1:1:1000
  %  T(j)=i;
   % rho(j)=z;
    %j=j+1;
   %end
%end


T_rho=XH1;

%x=0:.01:1;
yL_trained=trained(T_rho, Theta1L, Theta2L, Theta3L,Theta4L,Theta5L);
X_beta=[T_rho; yL_trained];
X_gamma=[T_rho; yL_trained];
h_Lin=trained_L(X_beta,Theta1Lin,Theta2Lin,Theta3Lin);
h_NL=trained_NL(X_gamma,Theta1NL,Theta2NL,Theta3NL);
h=(h_Lin+h_NL);



%figure()
%plot(yH1,h,'o')
%max= 1.7683834e-05 ;
Y=yH1;
H=h;
%min=  -1.3928500e-05;
Y=(maxH-minH)*yH1+minH;
H=(maxH-minH)*h+minH;
err=sum(abs((Y-H)./Y))/m*100
l=abs((Y-H)./Y);
error=median(l)
plot(Y,H,'o')
%hold on
%plot(XH,yH,'o')
%q=((6*x-2).^2).*sin(12*x-4);
%q=(x-sqrt(2)).*(sin(8*pi*x)).^2;
%q=high(x);
%plot(x,q)
%figure(2)
%plot(XL,yL,'*')
%p=.5*(((6*x-2).^2).*sin(12*x-4))+10*(x-.5)-5;
%p=sin(8*pi*x);
%p=low(x);
%hold on;
%plot(x,p)
%plot(x,yL_trained)
end
figure()
%filename3=sprintf('workspace_D2_%d.mat',mm);
filename3=sprintf('C:/15 Batch Thesis/Binary/Binary_P/workspace_P_%d_%d.mat',mm,ridzy);
save(filename3)
end
end

