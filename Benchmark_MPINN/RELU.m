function g = RELU(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
%g = piecewise(z>0, 1.0507009873554804934193349852946*z, z<=0,1.0507009873554804934193349852946*1.6732632423543772848170429916717*(exp(z)-1));

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

g(z>0)=z(z>0);
g(z<=0)=0;

end