function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

  % ====================== YOUR CODE HERE ======================
  % Instructions: Perform a single gradient step on the parameter
  % vector theta.
  %
  % Hint: While debugging, it can be useful to print out the values of
  % the cost function (computeCost) and gradient here.
  %
  alpha = 1e-2;
  size(X);
  size(y);
  size(theta);

  x0 = X'(1,:);
  x1 = X'(2,:);

  _t0  = theta(1) - alpha/m*x0*(X*theta - y);
  _t1  = theta(2) - alpha/m*x1*(X*theta - y);
  theta = [ _t0; _t1];

  % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
