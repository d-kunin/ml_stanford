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
H_TH = sigmoid(X*theta); % 100x1

T = -log(H_TH) .* y - log(1-H_TH) .* (1-y);
J = 1/m*sum(T);
grad = 1/m*((H_TH - y)'*X)';

if (false)
    fprintf("T\n")
    size(T)
    fprintf("y\n")
    size(y)
    fprintf("X\n")
    size(X)
    fprintf("H_TH\n")
    size(H_TH)
    fprintf("theta\n")
    size(theta)
    fprintf("grad\n")
    size(grad)
    fprintf("J\n")
    size(J)
endif
% =============================================================

end
