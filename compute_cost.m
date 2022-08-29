function cost = compute_cost(AL, Y)
%{
cost = COMPUTE_COST(AL,Y) compute the cross-entropy cost J with the
arguments; AL: probability vector corresponding to label predictions, shape
(1, number of examples) and Y: true "label" vector (for example: containing
0 if non-cat, 1 if cat). It returns cost -- the cross-entropy cost
%}

if nargin ~= 2
    error('Input error, check the function help for details on how to call the function')
end

[~,m] = size(Y);

% Compute cost
y1 = -Y .* log(AL);
y2 = (1 - Y) .* log(1-AL);
cost = (1/m) * (sum(sum((y1 - y2))));

