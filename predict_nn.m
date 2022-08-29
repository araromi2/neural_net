function p = predict_nn(X, y, parameters)
%{
This function is used to predict the result of a n-layer neural network.
It takes X: data set of examples we would like to label, parameters:
parameters of  the trained model and output p: predictions for the given
dataset X
%}
if nargin ~= 3
    error ('Input error, check the function help for more details')
end

[~,m] = size(X);
p = zeros(1,m);

% Forward propagation
[AL, ~] = model_fp(X,parameters);

% Convert probabilities to 0/1 predictions
[~,n] = size(AL);
for i = 1:n
    if AL(1,i) > 0.5
        p(1,i) = 1;
    else
        p(1,i) = 0;
    end
end

% Accuracy
fprintf('Accuracy: %.2f\n' , (mean(p(1,:) == y(1,:)))*100)
        