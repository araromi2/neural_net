function [grads,parameters] = model_nn(X, Y, layers_sizes, varargin)
%{
MODEL_NN(X,Y,layers_sizes,varagin) implements a L-layer neural network
takin agruments:
X: data, array of shape (no of features, number of examples)
Y: true "label" vector (containing 0 if cat, 1 if non-cat), of size
(1,number of examples)
layers_sizes: array containing the input size and each layer size, of
length (number of layer + 1)
Optional arguments:
learning_rate: learning rate of the gradient descent update rule
num_iterations: number of iterations of the optimization loop
print_cost: if True, it prints the cost every 100 steps
%}
p = inputParser;
addRequired(p, 'X');
addRequired(p, 'Y');
addRequired(p, 'layers_sizes');
addOptional(p,'learning_rate', 0.0075);
addOptional(p,'num_iterations', 8000);
addOptional(p,'print_cost', false, @islogical);
addOptional(p,'lambda', 0);
addOptional(p,'keep_prob', 1);

%parse argument
parse(p, X, Y, layers_sizes, varargin{:})
X = p.Results.X;
Y = p.Results.Y;
layers_sizes = p.Results.layers_sizes;
learning_rate = p.Results.learning_rate;
num_iterations = p.Results.num_iterations;
print_cost = p.Results.print_cost;
lambda = p.Results.lambda;
keep_prob = p.Results.keep_prob;

costs = [];

% Parameters initialization.
parameters = initialize_parameters(layers_sizes);

%gradient descent loop
for j = 1 : num_iterations
    %Forward propagation:
    if keep_prob == 1
        [AL, caches] = model_fp(X, parameters);
    elseif keep_prob < 1
        [AL, caches, dropout_cache] = model_fp_with_dropout(X, parameters, keep_prob);
    end
   %Compute cost.
   if lambda == 0
        cost = compute_cost(AL, Y);
   else
        cost = compute_cost_with_regularization(AL, Y, parameters, lambda);
   end
   %Backward propagation
   assert(lambda == 0 || keep_prob == 1)
   if lambda == 0 && keep_prob == 1
        grads = model_bp(X, parameters,AL, Y, caches);
   elseif keep_prob < 1
        grads = model_bp_with_dropout(X,parameters,AL, Y, caches, dropout_cache,keep_prob);
   elseif lambda ~= 0
       grads = model_bp_with_regularization(X,parameters,AL, Y, caches, lambda);
   end
   
   % Update parameters.
   parameters = update_parameters(parameters, grads, learning_rate);
   
   %Print the cost every 1000 training examples
   if print_cost && rem(j,1000) == 0
       fprintf ('Cost after iteration %i: %f\n', j ,cost)
   end
   if print_cost && rem(j,1000) == 0
       costs(end + 1) = cost;
   end
end
plot(costs)
ylabel('cost');
xlabel('iterations (per hundreds)')
title(strcat('Learning rate =',num2str(learning_rate)))


