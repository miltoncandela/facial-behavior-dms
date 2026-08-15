clc; clearvars;
df_train = readtable('validation_df2.csv');
df_test = readtable('new_validation_df2.csv');
% df = readtable('user_a01173858_I_L_facial_feature_data.csv');

%% extract features intro training and testing df

ta = df_train.ElapsedTime;
sa = df_train.CommandedState;
ma = [df_train.EARLeft df_train.EARRight df_train.EARAvg df_train.Blink ...
      df_train.MAR df_train.HeadRotX df_train.HeadRotY df_train.BlinkRate];

te = df_test.ElapsedTime;
se = df_test.CommandedState;
me = [df_test.EARLeft df_test.EARRight df_test.EARAvg df_test.Blink ...
      df_test.MAR df_test.HeadRotX df_test.HeadRotY df_test.BlinkRate];

colors = {'#a6cee3', '#1f78b4', '#fb9a99', '#b2df8a', '#33a02c'};
cmap = validatecolor(colors, 'multiple'); alpha = 0.5;
mksize = 20; seconds = 2.5; epoch_size = ceil(16.67*seconds);

%% boxplot of features across states
% Several decision thresholds (e.g., EAR < 50% baseline, MAR > 130% baseline, blink-rate thresholds) appear to be largely heuristic. A stronger statistical justification or sensitivity analysis would improve confidence in these design choices.
% confidence intervals, statistical significance tests, variance across runs, k-fold cross-validation

% declare calib_vals
calib_vals = [0.3199, 0.2334]; % EAR, MAR
calib_angles = [37.22 -70.32; -15.17 6.36];
jitterAmount = 0.2; capWidth = 0.2;

% blinkrate
y = ma(:,8); [y_ds, sa_ds] = mand_downsamp(sa, y, epoch_size);
[~, ~, stats] = kruskalwallis(y_ds, sa_ds, 'off'); % kruskal-wallis
c = multcompare(stats, 'CriticalValueType', 'bonferroni', 'Display', 'off');
x = sa_ds; N = length(x); x = x + (rand(N,1)-0.5)*2*jitterAmount;
fig = figure; hold on; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(x, y_ds, mksize, x, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha); ylabel('BlinkRate')
xticks([0 1 2 3 4]); yline(30, '--r','LineWidth',1); colormap(cmap); yticks(0:50:150)
plot_pvals(c,y_ds,ylim); ax = gca; ax.TickLength = [0.06 0.06];
exportgraphics(fig,'figs/boxBlinkRate_train.jpg', 'Resolution', 500)

% EAR
y = ma(:,3); [y_ds, sa_ds] = mand_downsamp(sa, y, epoch_size);
[~, ~, stats] = kruskalwallis(y_ds, sa_ds, 'off'); % kruskal-wallis
c = multcompare(stats, 'CriticalValueType', 'bonferroni', 'Display', 'off');
x = sa_ds; N = length(x); x = x + (rand(N,1)-0.5)*2*jitterAmount;
fig = figure; hold on; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(x, y_ds, mksize, x, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha); ylabel('EAR')
xticks([0 1 2 3 4]); yline(calib_vals(1), '--k','LineWidth',1); colormap(cmap); yticks(0:0.2:0.6)
yline(calib_vals(1)*0.5, '--r','LineWidth',1);
plot_pvals(c,y_ds,ylim); ax = gca; ax.TickLength = [0.06 0.06]; ylim([0 0.62])
exportgraphics(fig,'figs/boxEAR_train.jpg', 'Resolution', 500)

% MAR
y = ma(:,5); [y_ds, sa_ds] = mand_downsamp(sa, y, epoch_size);
[~, ~, stats] = kruskalwallis(y_ds, sa_ds, 'off'); % kruskal-wallis
c = multcompare(stats, 'CriticalValueType', 'bonferroni', 'Display', 'off');
x = sa_ds; N = length(x); x = x + (rand(N,1)-0.5)*2*jitterAmount;
fig = figure; hold on; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(x, y_ds, mksize, x, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha); ylabel('MAR'); 
xticks([0 1 2 3 4]); yline(calib_vals(2), '--k','LineWidth',1); colormap(cmap); yticks([0 0.5 1])
yline(calib_vals(2)*1.3, '--r','LineWidth',1);
plot_pvals(c,y_ds,ylim); ax = gca; ax.TickLength = [0.06 0.06];
exportgraphics(fig,'figs/boxMAR_train.jpg', 'Resolution', 500)

% HeadRot X
y = ma(:,6); [y_ds, sa_ds] = mand_downsamp(sa, y, epoch_size);
[~, ~, stats] = kruskalwallis(y_ds, sa_ds, 'off'); % kruskal-wallis
c = multcompare(stats, 'CriticalValueType', 'bonferroni', 'Display', 'off');
x = sa_ds; N = length(x); x = x + (rand(N,1)-0.5)*2*jitterAmount;
fig = figure; hold on; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(x, y_ds, mksize, x, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha); ylabel('HeadRot X'); 
xticks([0 1 2 3 4]); colormap(cmap); % yticks([-2 -1 0])
plot_pvals(c,y_ds,ylim); ax = gca; ax.TickLength = [0.06 0.06]; yticks(-100:50:100)
% yline(calib_angles(1,1), '--k'); yline(calib_angles(2,1), '--k'); % ylim([-110 110])
yline(calib_angles(1,1), 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1);
yline(calib_angles(2,1), 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1);
exportgraphics(fig,'figs/boxHeadrotX_train.jpg', 'Resolution', 500)

% HeadRot Y
y = ma(:,7); [y_ds, sa_ds] = mand_downsamp(sa, y, epoch_size);
[~, ~, stats] = kruskalwallis(y_ds, sa_ds, 'off'); % kruskal-wallis
c = multcompare(stats, 'CriticalValueType', 'bonferroni', 'Display', 'off');
x = sa_ds; N = length(x); x = x + (rand(N,1)-0.5)*2*jitterAmount;
fig = figure; hold on; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(x, y_ds, mksize, x, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha); ylabel('HeadRot Y'); 
xticks([0 1 2 3 4]); colormap(cmap); ylim([-90 90]); % yticks([-2 -1 0])
plot_pvals(c,y_ds,ylim); ax = gca; ax.TickLength = [0.06 0.06]; yticks(-100:50:100);
% yline(calib_angles(1,2), '--k'); yline(calib_angles(2,2), '--k'); 
yline(calib_angles(1,2), 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1);
yline(calib_angles(2,2), 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1);
exportgraphics(fig,'figs/boxHeadrotY_train.jpg', 'Resolution', 500) 

%% XY plots of features across states

% blinkrate for all states
% declare calib_vals
calib_vals = [0.3199, 0.2334]; % EAR, MAR

% training figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(ma(:,8), ma(:,3), mksize, sa, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha)
yline(calib_vals(1), '--k','LineWidth',1); 
yline(calib_vals(1)*0.5, '--r','LineWidth',1); xline(30, '--r','LineWidth',1);
xlabel('BlinkRate'); ylabel('EAR'); colormap(cmap)
ax = gca; ax.TickLength = [0.06 0.06]; xlim([-4 160]); ylim([0 1]);
exportgraphics(fig,'figs/EARBlink_train.jpg', 'Resolution', 500)

% testing figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(me(:,8), me(:,3), mksize, se, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha)
yline(calib_vals(1), '--k','LineWidth',1);
yline(calib_vals(1)*0.5, '--r','LineWidth',1); xline(30, '--r','LineWidth',1);
xlabel('BlinkRate'); ylabel('EAR'); colormap(cmap)
ax = gca; ax.TickLength = [0.06 0.06]; xlim([-4 160]); ylim([0 1]);
exportgraphics(fig,'figs/EARBlink_test.jpg', 'Resolution', 500) 

% MAR and EAR for all states, consdiering EAR calib and MAR calib
% declare calib_vals
calib_vals = [0.3199, 0.2334]; % EAR, MAR

% training figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(ma(:,5), ma(:,3), mksize, sa, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
xline(calib_vals(2), '--k','LineWidth',1); xline(calib_vals(2)*1.3, '--r','LineWidth',1); 
yline(calib_vals(1), '--k','LineWidth',1); yline(calib_vals(1)*0.5, '--r','LineWidth',1)
xlabel('MAR'); ylabel('EAR'); colormap(cmap)
ax = gca; ax.TickLength = [0.06 0.06]; ylim([0 1]); xlim([0 1.1])
exportgraphics(fig,'figs/EARMAR_train.jpg', 'Resolution', 500)

% testing figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2];
scatter(me(:,5), me(:,3), mksize, se, 'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
xline(calib_vals(2), '--k','LineWidth',1); xline(calib_vals(2)*1.3, '--r','LineWidth',1); 
yline(calib_vals(1), '--k','LineWidth',1); yline(calib_vals(1)*0.5, '--r','LineWidth',1)
xlabel('MAR'); ylabel('EAR'); colormap(cmap)
ax = gca; ax.TickLength = [0.06 0.06]; ylim([0 1]); xlim([0 1.1])
exportgraphics(fig,'figs/EARMAR_test.jpg', 'Resolution', 500)

% Head Rotation for all states
% calculate safe zone box
calib_angles = [37.22 -70.32; -15.17 6.36];
x = calib_angles(:,1); x_box = [x(1), x(2), x(2), x(1), x(1)];
y = calib_angles(:,2); y_box = [y(1), y(1), y(2), y(2), y(1)];

% training figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
scatter(ma(:,6), ma(:,7),mksize,sa,'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
plot(x_box, y_box, 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1); hold off; % draw safe zone

xlabel('HeadRotX (°)'); ylabel('HeadRotY (°)'); colormap(cmap);
xticks([-100 0 100]); yticks([-100 0 100]); xlim([-100 100]); ylim([-150 100])
ax = gca; ax.TickLength = [0.06 0.06];
exportgraphics(fig,'figs/headrot_train.jpg', 'Resolution', 500)

% testing figure
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
scatter(me(:,6),me(:,7),mksize,se,'filled', ...
    'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha);
plot(x_box, y_box, 'Color', validatecolor('#8B0000'), 'LineStyle', '--', 'LineWidth',1); hold off; % draw safe zone

xlabel('HeadRotX (°)'); ylabel('HeadRotY (°)'); colormap(cmap);
xticks([-100 0 100]); yticks([-100 0 100]); xlim([-100 100]); ylim([-150 100])
ax = gca; ax.TickLength = [0.06 0.06];
exportgraphics(fig,'figs/headrot_test.jpg', 'Resolution', 500)

%% OTHER

% colors = {'#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571'};
% colors = {'#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'};

% we need boxplots and stats to compare training from validation data

% not using HeadRotZ
figure; hold on;
plot(t, df.HeadRotX); plot(t, df.HeadRotY); plot(t, df.HeadRotZ); hold off;

% EARLeft and EARRight for EARAvg
colors = {'#a6cee3', '#1f78b4', '#fb9a99', '#b2df8a', '#33a02c'};
cmap = validatecolor(colors, 'multiple'); alpha = 0.5;
figure; scatter(m(:,1), m(:,2),30,m(:,3));
xlim([0 1.5]); ylim([0 1.5]); colorbar; colormap(cmap);

function [y_downsampled, sa_downsampled] = mand_downsamp(sa, y, epoch_size)
    states = unique(sa); nStates = length(states); 
    
    % Downsample each state individually to preserve block structure
    y_downsampled = []; sa_downsampled = [];
    for i_state = 1:nStates
        s = states(i_state);
        idx = find(sa == s); y_state = y(idx);
        num_epochs = floor(length(y_state) / epoch_size);
        
        state_means = zeros(num_epochs, 1);
        for e = 1:num_epochs
            start_idx = (e-1) * epoch_size + 1; end_idx = e * epoch_size;
            state_means(e) = mean(y_state(start_idx:end_idx));
        end
        
        y_downsampled = [y_downsampled; state_means];
        sa_downsampled = [sa_downsampled; ones(num_epochs, 1) * s];
    end
end

function plot_pvals(c,y_ds,y_limits)

    c(:,1) = c(:,1)-1; c(:,2) = c(:,2)-1; stests = [];
    for i = 1:size(c, 1)
        if c(i, 6) < 0.05
            groupA = c(i, 1); groupB = c(i, 2); span = abs(groupA - groupB); 
            stests = [stests; groupA, groupB, c(i, 6), span];
        end
    end
    if ~isempty(stests), stests = sortrows(stests, 4); end

    max_y = max(y_ds);
    space_step = (y_limits(2) - y_limits(1)) * 0.06; b_count = 0;
    
    % go through every stests
    for i = 1:size(stests, 1)
        groupA = stests(i, 1); groupB = stests(i, 2); p_pair = stests(i, 3);
        yb = max_y + (i*space_step*1.5); x1 = groupA; x2 = groupB;
        plot([x1, x1, x2, x2], ...
             [yb - space_step*0.4, yb, yb, yb - space_step*0.4], ...
             '-k', 'HandleVisibility', 'off');
         
        % determine the stars
        if p_pair < 0.001, stars = '***';
        elseif p_pair < 0.01, stars = '**';
        else, stars = '*'; end
        
        % place the text centered above the bracket line
        text(mean([x1, x2]), yb + space_step*0.2, stars, ...
             'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    end
    if ~isempty(stests), tb = size(stests, 1);
        ylim([y_limits(1), max_y + ((tb + 1) * space_step * 1.5)]); end
    hold off;
end