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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X * theta;
err = (h_theta - y).^2;
J_theta = 1 / (2*m) * sum(err);
reg = lambda / (2*m) * sum(theta(2:end).^2);
J = J_theta + reg;
% =========================================================================
mm = X * theta - y;
grad_theta =(1 / m * mm'*X);
reg = lambda / m * theta;
grad_theta = grad_theta(:) + reg;
grad_theta(1) = grad_theta(1) - lambda / m * theta(1);
grad = grad_theta;
grad = grad(:);

end
