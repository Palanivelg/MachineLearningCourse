function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
r = size(theta, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Un-regularized cost
h = (X * theta);
sqer = (h - y).^2;
J_no_reg = (sum(sqer) / (2*m));
%store theta1 as this should not be regularized
J_reg = lambda * (sum(theta .^2) - theta(1,1)^2)/(2*m);
J = J_no_reg + J_reg;

err = ((X * theta) - y);
grad(1) = (1/m) * sum(err .* X(:, 1));

for i = 2:r
grad(i) = (1/m) * sum(err .* X(:, i)) + (lambda/m) * theta(i);

% =========================================================================

grad = grad(:);

end
