function g = tanhGradient(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1-(tanh(z)).^2;
end
