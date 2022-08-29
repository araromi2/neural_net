function [A, cache] = linear_activation_fp(A_prev, W, b, activation)
%{
[A, cache] = LINEAR_ACTIVATION_FP(A_prev, W, b, activation) takes arguments
A_prev: activations from previous layer (or input data), W: weights matrix,
b: bias vector and activation ( which can either be 'sigmoid' or 'relu'
stored as a text  string. It returns A: the  output of the activation
function, also called the post-activation value and a cache: a MATLAB cell
containing "linear_cache" and "activation_cache"; stored for computing the
backward pass efficiently.
%}

if nargin ~= 4
    error('Input error, check the function help for details on how to call the function')
end

if strcmp(activation, 'sigmoid')
    [Z, linear_cache] = linear_fp(A_prev, W, b);      %Linear forward propagation
    [A, activation_cache] = sigmoid(Z);               %Activation function for the propagation
elseif strcmp(activation , 'relu')
    [Z, linear_cache] = linear_fp(A_prev, W, b);       %Linear forward propagation
    [A, activation_cache] = relu(Z);                   %Activation function for the propagation
end

cache = {linear_cache, activation_cache};

