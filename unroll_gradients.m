function grads = unroll_gradients(gradients)
grads = {};
for i = 1: (length(gradients)/2)-1
    dWI = gradients(strcat('dW', num2str(i)));
    dWI_unrolled = dWI(:);
    grads{end+1,1} = dWI_unrolled;
    dbI = gradients(strcat('db', num2str(i)));
    dbI_unrolled = dbI(:);
    grads{end+1,1} = dbI_unrolled;
end
grads = cell2mat(grads);