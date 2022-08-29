function[AL, caches] = model_fp(X, parameters)
%{
[AL, caches] = MODEL_FP(X, parameters) implement forward propagation
computation by taking X: example data, an array of shape(input size, number
of examples) and parameters: output of INITIALIZE_PARAMETERS() and
returning AL: las post-activation value (predictions) and caches: a MATLAB
cell containing every cache of LINEAR_ACTIVATION_FP() ( there are L of them
labeled from 1 to L)
%}

if nargin ~= 2
    error('Input error, check the function help for details on how to call the function')
end

caches = {};
A = X;
L = floor(length(parameters)/2);

for i = 1 : L-1
    A_prev = A;
    [A, cache] = linear_activation_fp(A_prev, parameters(strcat('W',num2str(i)))...
        ,parameters(strcat('b',num2str(i))), 'relu');
    caches{end+1} = cache;
end

%Linear-Sigmoid
[AL, cache] = linear_activation_fp(A, parameters(strcat('W',num2str(L)))...
        ,parameters(strcat('b',num2str(L))), 'sigmoid');
caches{end+1} = cache; 
        