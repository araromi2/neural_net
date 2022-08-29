function parameters_check = vector_to_map(theta, theta_map)
parameters_check = containers.Map;
for p = 1: length(theta_map)/2
    WI = theta_map(strcat('W',num2str(p)));
    [m,n] = size(WI);
    WI_unrolled = WI(:);
    
    bI = theta_map(strcat('b',num2str(p)));
    [o,q] = size(bI);
    bI_unrolled = bI(:);
    parameters_check(strcat('W',num2str(p))) = reshape(theta(1:length(WI_unrolled)), m,n);
    parameters_check(strcat('b',num2str(p))) = reshape(theta(1:length(bI_unrolled)), o,q); 
end