function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Carr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
Sarr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%Carr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%Sarr = [1];
%Carr = [0.01; 1; 3];
%Sarr = [0.01; 1; 3];
prErr = zeros(size(Carr,2), size(Carr,2));
x1 = X(:,1);
x2 = X(:,2);
for i=1:length(Carr)
   for j=1:length(Sarr)
       Ctmp = Carr(i);
       sigmaTmp = Sarr(j);
       model= svmTrain(X, y, Ctmp, @(x1, x2) gaussianKernel(x1, x2, sigmaTmp));
       predictions = svmPredict(model, Xval);
       prErr(i,j) = mean(double(predictions ~= yval));
   end
end
% =========================================================================
[mina,ind] = min(prErr(:));
[m,n] = ind2sub(size(prErr),ind);
C = Carr(m);
sigma = Sarr(n);
end
