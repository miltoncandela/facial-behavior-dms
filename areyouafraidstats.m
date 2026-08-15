clc; clearvars;
df = readtable('C:\Users\Milton\Documents\R\BRAIN-R\normmean_ICA_EEGc.csv');

% remove subject 1 and 2
% df = df{df{:,1} ~= 2 & df{:,1} ~= 1,:};

%%

subs = df{:,1}; usubs = unique(subs); nSubs = length(usubs);
scenes = df{:,2}; uscenes = unique(scenes); nScenes = length(uscenes);

nCols = size(df,2);

s = df{df.Scene>0 & df.Subject > 1,2};
for i_col = 3:nCols
    y = df{df.Scene>0 & df.Subject>1,i_col}; 
    % y = df{:,i_col};
    % y(y > 3) = 2.9; y(y < -3) = -2.9;
    y(y > 5) = 4.9; y(y < -5) = -4.9;
    y = reshape(y, nSubs-1, []);
    % [p, ~, stats] = kruskalwallis(y, s, 'off'); % kruskal-wallis
    [p, ~, stats] = friedman(y, 1); % kruskal-wallis
    disp(p)
    c = multcompare(stats);
    pvals = c(:,6)';
    % pvals = pvals*length(pvals);
    pvals = mafdr(pvals,'BHFDR',true);
    % disp(pvals)
    disp(pvals < 0.05)
    
    % disp([num2str(i_col), num2str(p < 0.05)])
    % disp(c(:,6)');
end

%% permutation testing and bonferroni correction with scene 0


m = zeros(nScenes-1, nCols-2);
for i_col = 3:nCols
    ybase = df{df.Scene == 0 & df.Subject>1,i_col};

    for i_scene = 2:nScenes
        y = df{df.Scene == i_scene-1 & df.Subject>1,i_col};
        [p, od, eff] = permutationTest(y,ybase,0,'exact',1);
        m(i_scene-1,i_col-2) = p;
    end
end

% update pvals according to Bonferronni
for i = 1:size(m,2)
    pvals = m(:,i)';
    disp(i)
    disp(pvals < 0.05)
    pvals = pvals*length(pvals); % bonferronni
    % pvals = mafdr(pvals,'BHFDR',true);
    % pvals = holmCorrection(pvals);
    % pvals = hochbergCorrection(pvals);
    m(:,i) = pvals;
    disp(pvals < 0.05)
end

writematrix(m,'EEG_pvaltopoplot.csv')

%% display each channel + bandpower corrected p-value
% according to latex table to have it automated

full_lobes = {'Frontal', 'Central', 'Parietal', 'Occipital'};
short_lobes = {'Fro', 'Cen', 'Par', 'Occ'};

% the first table of median and MAD values
nCols = size(df,2);
for i_col = 3:nCols
    colname = df.Properties.VariableNames(i_col); colname = colname{1};
    % if variable composed of Fro/Cen/Par/Occ, separate lobe and band
    % if not, then use it as it is (Engagement, Fatigue, Excitement)
    if i_col < 15
        words = split(colname, "_");
        full_lobe = full_lobes(find(ismember(short_lobes, words{1})));
        colname = ['$\\' lower(words{2}) '_\\text{' full_lobe{1} '}$']; end
    fprintf(colname);

    for i_scene = 1:nScenes
        y = df{df.Scene == i_scene-1 & df.Subject>1,i_col};
        % y(y > 6) = 6; y(y < -6) = -6;
        
        meanval = round(median(y),2); devval = round(mad(y, 1),2);
        fprintf([' & ' num2str(meanval) ' (' num2str(devval) ')']);
    end
    fprintf(' \\\\');
    if mod(i_col,3) == 2, fprintf('\\hline'); end
    fprintf('\n')
end

%%
disp('-----')

% the other table with inferential stats could be good with d and cliff
nCols = size(df,2);
for i_col = 3:nCols
    colname = df.Properties.VariableNames(i_col); colname = colname{1};
    % if variable composed of Fro/Cen/Par/Occ, separate lobe and band
    % if not, then use it as it is (Engagement, Fatigue, Excitement)
    if i_col < 15
        words = split(colname, "_");
        full_lobe = full_lobes(find(ismember(short_lobes, words{1})));
        colname = ['$\\' lower(words{2}) '_\\text{' full_lobe{1} '}$']; end
    fprintf(colname);

    ybase = df{df.Scene == 0 & df.Subject>1,i_col}; pvals = [];
    for i_scene = 2:nScenes
        y = df{df.Scene == i_scene-1 & df.Subject>1,i_col};
        [p, od, eff] = permutationTest(y,ybase,0,'exact',1);
        pvals = [pvals p];
    end
    pvals = pvals*length(pvals);
    % pvals = mafdr(pvals,'BHFDR',true); % BH
    % pvals = pvals*length(pvals);     % bonferonni

    for i_scene = 2:nScenes
        y = df{df.Scene == i_scene-1 & df.Subject>1,i_col};
        d = mean(y - ybase)/std(y - ybase);
        dif = y - ybase;
        delt = (sum(dif > 0) - sum(dif < 0)) / length(dif);
        p = round(pvals(i_scene-1),3); d = round(d,2); delt = round(delt,2);
        if p < 0.05, p = ['\\textbf{' num2str(p) '}']; else, p = num2str(p); end
        if abs(d) > 0.8, d = ['\\textbf{' num2str(d) '}']; else, d = num2str(d); end
        if abs(delt) > 0.47, delt = ['\\textbf{' num2str(delt) '}']; else, delt = num2str(delt); end
        fprintf([' & ' p ' / ' d ' / ' delt]);
    end
    
    fprintf(' \\\\');
    if mod(i_col,3) == 2, fprintf('\\hline'); end
    fprintf('\n')
end

% $\beta_\text{Central}$ & 0.69 / 0.49 / 0.12 & 0.78 / 0.60 / 0.12 & 0.78 / 0.78 / 0.12 & 0.59 / 0.38 / -0.12 \\


%%


%%
i_col = 16; % 3+4; % i_col = 9;

y = df{:,i_col}; x = df{:,2}; c = df{:,1};
y(y > 3) = 2.9; y(y < -3) = -2.9;

% figure; 
% scatter(x,y,15,c); ylim([-3 3])

xu = unique(x);

ym = zeros(size(xu));
ys = zeros(size(xu));

for i = 1:length(xu)
    idx = x == xu(i);
    ym(i) = mean(y(idx));
    ys(i) = std(y(idx),0,1)/sqrt(8);
end

figure;
scatter(x, y, 15, c, 'filled')
hold on
errorbar(xu, ym, ys, 'k', 'LineStyle', 'none', 'LineWidth', 1.5)
hold off
% errorbar(xu, ym, ys, 'ko-', 'LineWidth', 1.5)
xlabel('x')
ylabel('Mean y')

figure;
bar(xu, ym)
hold on

capWidth = 0.15;   % Width of the cap in x-units

%% lets do a rough heatmap of responses

nCols = size(df,2); 
scenes = df{:,2}; uscenes = unique(scenes); nScenes = length(uscenes);
m = zeros(nScenes, nCols-2);
for i_col = 3:nCols
    for i_scene = 1:nScenes
        y = df{df.Scene == i_scene-1,i_col};
        % y(y > 3) = 2.9; y(y < -3) = -2.9;
        m(i_scene,i_col-2) = median(y);
    end
end

figure; heatmap(m); clim([-1 1]); colormap jet

%%
for i = 1:numel(xu)
    % Vertical line
    plot([xu(i) xu(i)], [ym(i) ym(i)+ys(i)], 'k', 'LineWidth', 1.5)

    % Top cap only
    plot([xu(i)-capWidth xu(i)+capWidth], ...
         [ym(i)+ys(i) ym(i)+ys(i)], ...
         'k', 'LineWidth', 1.5)
end

hold off

function p_holm = holmCorrection(p)

m = numel(p);
[p_sorted,idx] = sort(p);

p_adj = (m-(1:m)+1).*p_sorted;

% enforce monotonicity
for i = 2:m
    p_adj(i) = max(p_adj(i),p_adj(i-1));
end

p_adj = min(p_adj,1);

p_holm = zeros(size(p));
p_holm(idx) = p_adj;

end

function p_hoch = hochbergCorrection(p)

m = numel(p);
[p_sorted,idx] = sort(p,'descend');

p_adj = (1:m).*p_sorted;

for i = 2:m
    p_adj(i) = min(p_adj(i),p_adj(i-1));
end

p_adj = min(p_adj,1);

p_hoch = zeros(size(p));
p_hoch(idx) = p_adj;

end