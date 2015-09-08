function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
for itr = 1:m
    h_theta = sigmoid(sum(transpose(theta).*X(itr,:)));
    J_theta = -1*y(itr)*log(h_theta)-(1-y(itr))*log(1-h_theta);
    J = J + J_theta;
end
grad_theta = grad;
for itr = 1:m
    h_theta = sigmoid(sum(transpose(theta).*X(itr,:)));
    for j = 1:length(grad)
       grad_theta(j) = (h_theta - y(itr))*X(itr,j);         
    end
    grad = grad + grad_theta;
end
% =============================================================
theta_J = 0;
for j = 2:length(grad)
    theta_J = theta_J + theta(j)^2;
end
theta_J = theta_J*lambda/(2*m);
J = J / m + theta_J;
grad = grad / m + [0; (lambda/m)*theta(2:end)];
end
