%%Name: Chao Fan
% Pid: A53078965

%%read data, seperate feature and label, set size.
M_train = dlmread('hw2train.txt');
M_valid = dlmread('hw2validate.txt');
M_test = dlmread('hw2test.txt');

F_train = M_train(:, 1 : end -1);
n_F = size(F_train , 1);
F_valid = M_valid(:, 1 : end -1);
n_v_F = size(F_valid , 1);
F_test = M_test(:, 1 : end -1);
n_t_F = size(F_test , 1);

L_train = M_train(:, end);
P_train = zeros(n_F,1);
L_valid = M_valid(:, end);
P_valid = zeros(n_v_F,1);
L_test = M_test(:, end);
P_test = zeros(n_t_F,1);

%%
%%question3.1 build k-nn classifer from train data and
%write down a table of trainning errors.
d_error_matrix = zeros(n_F, n_F);
for i = 1 : n_F
    for j = 1: n_F
        d_error_matrix(i,j) = norm(F_train(i,:) - F_train(j,:));
    end
end

[~,sort_index] = sort(d_error_matrix);

K = [1 3 5 11 16 21];

display(['question3.1 part1: K-NN from train. Table of training errors']);

for k = 1 : length(K)
    for L = 1 : n_F
        M_sort_vector = sort_index(1: K(k), L);
        P_train(L, 1) = count_freq(M_sort_vector, L_train);
    end
    error_train = 1 - sum(P_train == L_train)/n_F;
    display(['k=', num2str(K(k)), ' error_train=', num2str(error_train)]);
end
%%
%%question3.1 build k-nn classifer from train data and
%write down a table of validation errors.


d_error_matrix_t_v = zeros(n_F, n_v_F);
for i = 1 : n_F
    for j = 1: n_v_F
        d_error_matrix_t_v(i,j) = norm(F_train(i,:) - F_valid(j,:));
    end
end

[~,sort_index_t] = sort(d_error_matrix_t_v);

display(['question3.1 part2: K-NN from train. Table of validate errors']);

K = [1 3 5 11 16 21];

for k = 1 : length(K)
    for L = 1 : n_v_F
        M_sort_vector = sort_index_t(1: K(k), L);
        P_valid(L, 1) = count_freq(M_sort_vector, L_train);
    end
    error_train = 1 - sum(P_valid == L_valid)/n_v_F;
    display(['k=', num2str(K(k)), ' error_valid=', num2str(error_train)]);
end

%%
%question3.1 build k-nn classifer from train data and
%Test error of this classifier(here we choose 1-nn classifier)
d_error_matrix_test_v = zeros(n_F, n_t_F);
for i = 1 : n_F
    for j = 1: n_t_F
        d_error_matrix_test_v(i,j) = norm(F_train(i,:) - F_test(j,:));
    end
end

[~,sort_index_test] = sort(d_error_matrix_test_v);

display('question3.1 part3: Test error of 1-NN classifier');

for L = 1 : n_v_F
    M_sort_vector = sort_index_test(1, L);
    P_test(L, 1) = count_freq(M_sort_vector, L_train); 
end
error_test = 1 - sum(P_test == L_test)/n_t_F;
display(['k=1, error_test= ', num2str(error_test)]);

%%
%choose 3-NN classifier and compute the confusion matrix
d_error_matrix_test_v1 = zeros(n_F, n_t_F);
for i = 1 : n_F
    for j = 1: n_t_F
        d_error_matrix_test_v1(i,j) = norm(F_train(i,:) - F_test(j,:));
    end
end

[~,sort_index_test1] = sort(d_error_matrix_test_v1);

display('question3.2 Test error of 1-NN classifier');

for L = 1 : n_v_F
    M_sort_vector1 = sort_index_test1(1: 3, L);
    P_test(L, 1) = count_freq(M_sort_vector1, L_train);  
end
error_test = 1 - sum(P_test == L_test)/n_t_F;
display(['k=3, error_test= ', num2str(error_test)]);
Confusion_matrix = zeros(10,10);

[Confusion_matrix,order] = confusionmat(M_test(:, end),P_test);
Confusion_matrix = Confusion_matrix'./repmat(sum(Confusion_matrix'),10,1);
Confusion_matrix