function parameters = update_parameters(parameters, grads, learning_rate)
%{
UPDATE_PARAMETERS(parameters, grads, learning_rate) update parameters using
gradient descent using parameters: a MATLAB map containing parameters,
grads: MATLAB map containing gradients, output of model_bp.
It returns parameters: a MATLAB map containing the updated parameters.
%}

if nargin ~= 3
    error('Input error, check the function help for details on how to call the function')
end

L = floor(length(parameters)/2);

for i = 1:L
    parameters(strcat('W',num2str(i))) = parameters(strcat('W',num2str(i)))...
        - learning_rate .* (grads(strcat('dW',num2str(i))));
    parameters(strcat('b',num2str(i))) = parameters(strcat('b',num2str(i)))...
        - learning_rate .* (grads(strcat('db',num2str(i))));
end
