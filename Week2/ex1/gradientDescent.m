function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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

    derivative = X * theta; 
    derivative = derivative - y;
    temporary1 = sum(derivative.* X(:,1));
    temporary2 = sum(derivative.* X(:,2));
    temporary1 = temporary1/m;
    temporary1 = temporary1* alpha;
    temporary2 = temporary2/m;
    temporary2 = temporary2*alpha;
    temporary1 = theta(1) - temporary1;
    temporary2 = theta(2) - temporary2;

    theta(1) = temporary1;
    theta(2) = temporary2;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
