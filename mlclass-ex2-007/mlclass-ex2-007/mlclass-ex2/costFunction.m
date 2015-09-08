function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
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
grad = grad / m;
J = J / m;
end
