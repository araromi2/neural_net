function cost = compute_cost_with_regularization(AL, Y, parameters, lambda)
%{
Implement the cost function with L2 regularization. It take arguments:
AL - post-activation, output of forward propagation of size(output size,number of examples)
Y - "true" labes vector, of shape (output size, number of examples)
parameters - MATLAB map containing parameters of the model.
It returns cost - value of the regularized loss function
%}

if nargin ~= 2
    error('Input error, check the function help for details on how to call the function')
end
[~,m] = size(Y);
SW = zeros(1,length(parameters)/2);

cross_entropy_cost = compute_cost(AL,Y);

for i = 1: length(parameters)/2
    SW(i) = sum(sum(parameters(strcat('W',num2str(i))) .^ 2));
end

L2_regularization_cost = (lambda/(2*m)) .* (sum(SW));

cost = cross_entropy_cost + L2_regularization_cost;