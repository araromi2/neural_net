function [dA_prev, dW, db] =  linear_activation_bp(X,Y, parameters,dA, cache, activation)
%{
LINEAR_ACTIVATION_BP(dA, cache, activation) implement the backward
propagation for a layer. It takes the arguments: dA: post-activation
gradient for the current layer, cache: cell of values (linear_cache,
activation_cache) we store for computing backward pass efficiently.It
returns dA_prev: gradient of the cost with respect ton the activation of
the previous layer, dW: gradient of the cost with respect to W of the
current layer (same size as W) and db: gradient of the cost with resepct to
b of the current layer (same size as b)
%}

if nargin ~= 6
    error('Input error, check the function help for details on how to call the function')
end

linear_cache = cache{1}; activation_cache = cache{2};

if strcmp(activation, 'relu')
    dZ = relu_bp(dA, activation_cache);
    [dA_prev, dW, db] = linear_bp(dZ, linear_cache);
elseif strcmp(activation, 'sigmoid')
    dZ = sigmoid_bp(X,Y, parameters);
    [dA_prev, dW, db] = linear_bp(dZ, linear_cache);
end