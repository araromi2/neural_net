function [Z, cache] = linear_fp(A, W, b)
%{
[Z,cache] = LINEAR_FP(A, W, b) implement a layer's forward propagation.
It takes A, activation from previous layer (or input data): (size of
previous layer, number of examples),W: weight array of shape(size of
current layer, size of pevious layer) and b: bias array of shape(size of
the current layer, 1) and return Z: the  input of the activation function,
also called pre_activation parameter and cache: a MATLAB cell containing
"A", "W" and "b"; stored for computing the backward pass efficiently.
%}

if nargin < 3
    error('linear_fp require A,W,b. Check the help for the function for more details.')
end
Z = bsxfun(@plus, (W*A), b);
cache = {A, W, b};