%%Name: Chao Fan
% Pid: A53078965
% HW2 Problem3 MATLAB CODE 

function pred = count_freq(M_sort_vector, L_train)

siz = size(M_sort_vector, 1);
pred_v = zeros(siz, 1);

for i = 1 : siz
    index = M_sort_vector(i);
    pred_v(i,1) = L_train(index);
end

pred = mode(pred_v);