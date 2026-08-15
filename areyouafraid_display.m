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
        
        meanval = round(median(y),2); devval = round(mad(y, 1),2);
        fprintf([' & ' num2str(meanval) ' (' num2str(devval) ')']);
    end
    fprintf(' \\\\');
    if mod(i_col,3) == 2, fprintf('\\hline'); end
    fprintf('\n')
end

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
    pvals = mafdr(pvals,'BHFDR',true); % BH
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
