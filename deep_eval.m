clc; clearvars;
df_train = readtable('validation_df2.csv');
df_test = readtable('new_validation_df2.csv');

pred_train = readtable('RNN_train.csv'); pred_train(:, 1) = [];
pred_test = readtable('RNN_test.csv'); pred_test(:, 1) = [];
pred_train = table2array(pred_train); pred_test = table2array(pred_test);



%%



ytrain = df_train{:, 'CommandedState'}; ytest = df_test{:, 'CommandedState'};
[~, row_argmax] = max(pred_train,[],2); ytrain_pred = row_argmax-1;
[~, row_argmax] = max(pred_test,[],2); ytest_pred = row_argmax-1;

% align both
seq_size = 5;
ytrain = ytrain(seq_size+1:end,:); ytest = ytest(seq_size+1:end,:);
