function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h_theta = 0;
    J1_theta = 0;
    J2_theta = 0;   
    for i = 1:m
        h_theta = theta(1)*1.0+theta(2)*X(i,2);
        J1_theta = J1_theta+(h_theta-y(i))*X(i,1);
        J2_theta = J2_theta+(h_theta-y(i))*X(i,2);
    end    
    theta(1) = theta(1)-alpha*(1/m)*J1_theta;
    theta(2) = theta(2)-alpha*(1/m)*J2_theta;
    
    
    
    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end
    
end
