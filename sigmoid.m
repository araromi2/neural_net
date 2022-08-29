function [A, activation_cache] = sigmoid(Z)
%{
[A, activation_cache] = SIGMOID(Z) takes the input of the activation
function  (or preactivation parameters) and return A: the sigmoid of the Z
and activation_cache; a MATLAB cell containing Z which is useful in
computing the  backward propagation for the activation.
NB: Z can be a number or array.
%}

% Make sure there is at least one input
if nargin < 1 || nargin > 1
    error('sigmoid only take one input')
end

A = 1 ./ (1 + exp(-Z));
activation_cache = {Z};