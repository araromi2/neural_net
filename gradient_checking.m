function gradient_checking(parameters, gradients, X, Y)
%{
Check if backward propagation computes correctly the gradient of the cost
output by forward propagation
Arguments:
parameters: MATLAB map containing trained parameters "W1", "b1" ...
grad: output of backward propagation dW1, db1, ...
X: input datapoint, of shape (input size,m)
Y: true "label"
%}

if nargin < 4 
    error('Input error, check the function help for more details')
end
epsilon = 1e-7;
nn_params = unroll_parameters(parameters);
gradient = unroll_gradients(gradients);
J_plus = zeros(size(nn_params));
J_minus = zeros(size(nn_params));
gradapprox = zeros(size(nn_params));
for p = 1:numel(nn_params)
    thetaplus = nn_params;
    thetaplus(p) = thetaplus(p) + epsilon;
    [AL, ~] = model_fp(X, vector_to_map(thetaplus,parameters));
    J_plus(p) = compute_cost(AL, Y);
    
    thetaminus = nn_params;
    thetaminus(p) = thetaminus(p) - epsilon;
    [AL, ~] = model_fp(X, vector_to_map(thetaminus, parameters));
    J_minus(p) = compute_cost(AL, Y);
    
    gradapprox(p) = (J_plus(p) - J_minus(p))/ (2 * epsilon);
end

disp([gradapprox gradient]);

numerator = norm(gradient - gradapprox);
denominator = norm(gradapprox) + norm(gradient);
difference = numerator/denominator;
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 2e-7). \n' ...
         '\nRelative Difference: %g\n'], difference);
if difference > 2e-7
    fprintf ('There is a mistake in the backward propagation! difference = %g \n',difference)
else
    fprintf ('Backward propagation works perfectly fine! difference = %g \n', difference)
end
            