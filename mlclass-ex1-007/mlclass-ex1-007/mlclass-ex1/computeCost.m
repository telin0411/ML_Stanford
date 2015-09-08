function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J_tmp = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for itr = 1:m
    h_theta = theta(1)*1.0+theta(2)*X(itr,2);
    J_theta = (h_theta-y(itr))^2;
    J_tmp = J_tmp+J_theta;        
end
% =========================================================================
J_tmp = (J_tmp/2)/m;
J = J_tmp
end
