function [x,y,parameters] = gradient_check_test_case()
rng (1)
parameters = containers.Map;
x = randn(4,3);
y = [1,1,0];
parameters('W1') = randn(5,4);
parameters('b1') = randn(5,1);
parameters('W2') = randn(3,5);
parameters('b2') = randn(3,1);
parameters('W3') = randn(1,3);
parameters('b3') = randn(1,1);

