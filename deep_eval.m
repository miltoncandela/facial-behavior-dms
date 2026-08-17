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


%% instead of evaluating accuracy, we can plot training and validation loss
% curves by varying sequence length (preferably average across seeds)

model_type = 'GRU'; epoch = 30; batch_size = 128; neurons = 8;
seqs = [5 10 20 40 80 120]; nSeqs = length(seqs);
cmap = copper(nSeqs);
seeds = 1:10; nSeeds = length(seeds);
% cmap = brewermap(nSeqs, 'Blues');
% cmap = brewermap(nSeqs, 'Accent');

fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
for i_seq = 1:nSeqs
    seq = seqs(i_seq);

    mall = zeros(epoch, 2, nSeeds);

    for seed = seeds
        folder = ['seqGRU/' model_type '_E' num2str(epoch) '_B' num2str(batch_size) '_NE' num2str(neurons) '_S' num2str(seq) '_SE' num2str(seed)];
        df = readtable([folder '/csv/history.csv']);
        
        m = df{:,{'loss', 'val_loss'}};
        mall(:,:,seed) = m;
    end
    m = squeeze(mean(mall,3)); ms = squeeze(std(mall,[],3));
    ci = (ms/sqrt(nSeeds))*1.96;
    mhigh = m + ci; mlow = m - ci;
    evec = 1:epoch;
    plot(evec, m(:,1), 'LineStyle', '-', 'Color', cmap(i_seq,:), 'DisplayName', num2str(seq));
    fill([evec fliplr(evec)], [mlow(:,1)' fliplr(mhigh(:,1)')], cmap(i_seq,:), 'FaceAlpha', 0.2, 'EdgeColor', 'None', 'HandleVisibility','off')
    plot(evec, m(:,2), 'LineStyle', '--', 'Color', cmap(i_seq,:), 'HandleVisibility','off');
    fill([evec fliplr(evec)], [mlow(:,2)' fliplr(mhigh(:,2)')], cmap(i_seq,:), 'FaceAlpha', 0.2, 'EdgeColor', 'None', 'HandleVisibility','off')
end
hold off;
exportgraphics(fig,['figs/' model_type '_seq.jpg'], 'Resolution', 500)