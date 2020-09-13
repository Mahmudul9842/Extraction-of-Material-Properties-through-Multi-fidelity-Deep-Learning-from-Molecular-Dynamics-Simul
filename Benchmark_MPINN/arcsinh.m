function g = arcsinh(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = log(z+sqrt(z.^2+1));
end
