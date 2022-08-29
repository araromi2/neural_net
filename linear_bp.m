function [dA_prev, dW, db] = linear_bp(dZ, cache)
%{
LINEAR_BP(dZ, cache) implement the linear portion of backward propagation
for a single layer with arguments dZ: gradient of the cost with respect
to the linear output and cache: cell of values (A_prev, W, b) coming from
the forward propagation. It returns dA_prev: Gradient of the cost with
respect to the activation (of the previous layer), dW: Gradient of the cost
with respect to W(current layer l), same shape as W and db: Gradient of the
cost with respect to b (current layer l) of the same size as b.
%}

if nargin ~= 2
    error('Input error, check the function help for details on how to call the function')
end

A_prev = cache{1}; W = cache{2}; b = cache{3};
[~,m] = size(A_prev);

dW = (1/m) .* (dZ * A_prev');
db = (1/m) .* (sum(dZ,2));
dA_prev = W' * dZ;
