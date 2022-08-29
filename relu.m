function [A, activation_cache] = relu(Z)
%{
[A, activation_cache] = relu(Z) takes the input of the activation
function  (or preactivation parameters) and return A: the relu of Z
and activation_cache; a MATLAB cell containing Z which is useful in
computing the  backward propagation for the activation.
NB: Z can be a number or array.
%}

% Make sure there is at least one input
if nargin < 1 || nargin > 1
    error('relu only take one input')
end

A = max(0,Z);
activation_cache = {Z};
