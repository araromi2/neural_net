function parameters = initialize_parameters(layer_size)
% parameters = INITIALIZE_PARAMETERS(layer_size) take in the argument
% layer_size which is a vector containng the size(dimension) of each 
% layer and returns PARAMETERS which is a MATLAB map containing parameters
% 'Wl', 'bl'..., 'WL', 'bL'
% Wl is a weight matrix of shape(layer_size(l+1),layer_size(l)) and bl is a
% bias vector of shape (layer_size(l+1),1)

% Make sure there is at least one input
if nargin < 1 || nargin > 1
    error('initialize_parameters only take one input')
end

%Make sure the output is the number required
if nargout >1
    error('Too much output')
end

parameters = containers.Map;
L = length(layer_size);
rng(1);
%Initialize parameter using HE initialization
for i = 1 : L-1
    parameters(strcat('W',num2str(i))) = randn(layer_size(i+1),layer_size(i))*sqrt(2./layer_size(i));
    parameters(strcat('b',num2str(i))) = zeros(layer_size(i+1),1);
end

