function g = RELUGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).



%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
%g = piecewise(z>0, 1.0507009873554804934193349852946*z, z<=0,1.0507009873554804934193349852946*1.6732632423543772848170429916717*(exp(z)-1));

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

g = zeros(size(z));
g(z>0)=1;
g(z<=0)=0;

% =============================================================

end