function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = (2.0 ./ (1.0 + exp(-2*z)))-1;
end
