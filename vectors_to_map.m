function parameters = vectors_to_map(theta)
parameters = containers.Map;
parameters(strcat('W',num2str(1))) = reshape(theta(1:6), 3,2);
parameters(strcat('b',num2str(1))) = reshape(theta(7:9), 3,1);
parameters(strcat('W',num2str(2))) = reshape(theta(10:12), 1,3);
parameters(strcat('b',num2str(2))) = reshape(theta(13:end), 1,1);
end