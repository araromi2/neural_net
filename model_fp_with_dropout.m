function[AL, caches, dropout_cache] = model_fp_with_dropout(X, parameters, keep_prob)
%{
Implement the forward propagation with dropout regularization. 
Arguments: X: input datatset of size (number of features, number of
examples)
parameters: MATLAB map containing the parameters; W1,b1, ...
keep_prob : probability of keeping a neuron active during dropout, scalar

Returns:
AL: last activation value, output of the forward propagation
cache: MATLAB cell, information stored for computing the backward prop.
%}

if nargin ~= 3
    error('Input error, check the function help for details on how to call the function')
end

caches = {};
dropout_cache = {};
A = X;
L = floor(length(parameters)/2);

for i = 1 : L-1
    A_prev = A;
    [A, cache] = linear_activation_fp(A_prev, parameters(strcat('W',num2str(i)))...
        ,parameters(strcat('b',num2str(i))), 'relu');
    D = rand(size(A));     % initialize matrix D
    D = D < keep_prob;      % convert entries of D to 0 and 1 (using keep_prob as the threshold)
    A = A .* D;              % shut down some neurons of A
    A = A ./ keep_prob;       % scale the value of neurons that haven't been shut down
    dropout_cache{end+1} = D;
    caches{end+1} = cache;
end

%Linear-Sigmoid
[AL, cache] = linear_activation_fp(A, parameters(strcat('W',num2str(L)))...
        ,parameters(strcat('b',num2str(L))), 'sigmoid');
caches{end+1} = cache;