function nn_params = unroll_parameters(parameters)
nn_params = {};
for i = 1: length(parameters)/2
    WI = parameters(strcat('W', num2str(i)));
    WI_unrolled = (WI(:));
    nn_params{end+1,1} = WI_unrolled;
    bI = parameters(strcat('b', num2str(i)));
    bI_unrolled = (bI(:));
    nn_params{end+1,1} = bI_unrolled;
end
nn_params = cell2mat(nn_params);