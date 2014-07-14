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

H_X = sigmoid(X*theta);
sub_theta = theta(2:end);

J = 1/m*sum((-y .* log(H_X) - (1 - y) .* log(1 - H_X))) + lambda/(2*m)*sum(sub_theta .^ 2);
    
% grad
grad(1) = 1/m*sum((H_X - y) .* X(:,1));

for i = 2:size(X)(2)
   grad(i) = 1/m*(H_X - y)' * X(:,i) + lambda/m*theta(i); 
end

% logging
if (false) 
    printf("theta\n");
    size(theta)
    printf("sub_theta\n");
    size(sub_theta)
    printf("X\n");
    size(X)
    printf("y\n");
    size(y)
    printf("lambda\n");
    size(lambda)
    printf("H_X\n");
    size(H_X)
    printf("J\n");
    size(J)
    printf("grad\n");
    size(grad)
endif



% =============================================================

end
