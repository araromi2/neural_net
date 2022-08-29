function grads = model_bp_with_regularization(X,parameters,AL, Y, caches, lambda)
%{
MODEL_BP(AL,Y,caches) implement the backward propagation for all layers. It
takes AL: probability vector, output of the forward propagation, Y: true
"label" vector (containing 0 if non-cat, 1 if cat), caches: MATLAB cell of
caches containing every cache of LINEAR_ACTIVATION_FP() with "relu" (it is
caches{l} for l in range (L-1) i.e. l = 1...L-1, the cache of
LINEAR_ACTIVATION_FP() with "sigmoid" (it's caches{L}), parameters of the
model and the regularization  parameter lambda. it returns the gradient
%}

if nargin ~= 5
    error('Input error, check the function help for details on how to call the function')
end

grads = containers.Map;
L = length(caches);        %number of layers
[~,~] = size(AL);
Y = reshape(Y,size(AL)); %to make sure Y is 1-by-m

%Initialize the backpropagation
dAL = (Y ./ AL) - ((1 - Y)./(1 - AL));


%Output layer
current_cache = caches{L};
[dA_prevL, dWL, dbL] = linear_activation_bp(X,Y,parameters,dAL, current_cache, 'sigmoid');
grads(strcat('dA',num2str(L-1))) = dA_prevL;
grads(strcat('dW',num2str(L))) = dWL + (lambda/m) .* ...
    parameters(strcat('W', num2str(L)));
grads(strcat('db',num2str(L))) = dbL;

for i = L-1:-1:1
    current_cache = caches{i};
    [dA_prev_temp, dW_temp, db_temp] = linear_activation_bp(X,Y,parameters,grads(strcat('dA',num2str(i))), current_cache, 'relu');
    grads(strcat('dA',num2str(i-1))) = dA_prev_temp;
    grads(strcat('dW',num2str(i))) = dW_temp + (lambda/m) .* ...
    parameters(strcat('W', num2str(i)));
    grads(strcat('db',num2str(i))) = db_temp;
end;