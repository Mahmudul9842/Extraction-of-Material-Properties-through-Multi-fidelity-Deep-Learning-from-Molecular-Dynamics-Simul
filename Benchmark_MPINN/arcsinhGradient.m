function g = arcsinhGradient(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ sqrt(z.^2+1);
end
