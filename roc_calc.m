clc; clearvars;
df = readtable('skfold_res10.csv');

folds = unique(df.fold); nFolds = length(folds);
classes = unique(df.class); nClasses = length(classes);
colors = {'#a6cee3', '#1f78b4', '#fb9a99', '#b2df8a', '#33a02c'};
colors2 = zeros(3,length(colors));
for i_color = 1:length(colors), colors2(:,i_color) = hex2rgb(colors{i_color}); end
colors = colors2'; alpha = 1;

mauc = zeros(nClasses, nFolds);
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
for i_class = 1:nClasses
    
    class = classes(i_class); cdf = df(df.class == class,:);
    N = 100; m = zeros(N,nFolds); meanFPR = linspace(0,1,N);    
    
    for i_fold = 1:nFolds
        fold = folds(i_fold); ccdf = cdf(cdf.fold == fold,:);

        % interpolate TRP based on template FPR
        [fpr_unique,~,ic] = unique(ccdf.fpr);
        tpr_unique = accumarray(ic,ccdf.tpr,[],@max);
        interp_tpr = interp1(fpr_unique,tpr_unique,meanFPR,'linear');
        interp_tpr(1) = 0; interp_tpr(end) = 1;

        % store on matrix
        m(:,i_fold) = interp_tpr;
        mauc(i_class, i_fold) = trapz(interp_tpr);
    end
    meanv = mean(m,2); stdv = std(m,0,2); CI95 = 1.96*stdv/sqrt(nFolds);
    highv = meanv + CI95; lowv = meanv - CI95;
    
    plot(meanFPR, meanv, 'Color', [colors(i_class,:) alpha], 'LineWidth', 1.5)
    fill([meanFPR fliplr(meanFPR)], [lowv' fliplr(highv')], colors(i_class,:), 'faceAlpha', 0.2, 'EdgeColor', 'None', 'HandleVisibility', 'off')
end
plot(meanFPR, meanFPR, '--k')
xlim([0 1]); ylim([0 1]); xticks([0 0.5 1]); yticks([0 0.5 1]);
xlabel('False Positive Rate (FPR)'); ylabel('True Positive Rate (TPR)')
ax = gca; ax.TickLength = [0.06 0.06]; hold off;
% exportgraphics(fig,['figs/roc' num2str(nFolds) '.jpg'], 'Resolution', 500)

%%

mksize = 20; alpha = 0.5;
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
% for i_class = 1:nClasses
    % scatter(i_class, mauc(i_class,:), mksize, colors(i_class,:), ...
    %         'filled', 'MarkerFaceAlpha', alpha, 'MarkerEdgeAlpha', alpha)
% end

m = mean(mauc,2); disp(m)
se = std(mauc,[],2)/sqrt(nFolds); ci95 = 1.96*se;

b = bar(1:nClasses,m); b.FaceColor = 'flat';
for i = 1:nClasses, b.CData(i,:) = colors(i,:); end
errorbar(1:nClasses,m,0,ci95,'k','LineStyle', 'none')

ylim([85 100]); xlabel('Class'); ylabel('AUC (%)'); hold off;
ax = gca; ax.TickLength = [0.06 0.06]; xlim([0 6])
% exportgraphics(fig,['figs/rocauc' num2str(nFolds) '.jpg'], 'Resolution', 500)

%%

clc; clearvars;
df = readtable('skfold_acccomp10.csv');

folds = df.fold; nFolds = length(folds);
colors = {'#cccccc', '#525252'};
cmap = validatecolor(colors, 'multiple');

macc = [df.accH df.accM df.f1H df.f1M]';
m = mean(macc,2);
se = std(macc,[],2)/sqrt(nFolds); ci95 = 1.96*se;
m = [m(1:2,:) m(3:4,:)];
ci95 = [ci95(1:2,:) ci95(3:4,:)];

nClasses = 2; 
fig = figure; fig.Units = 'inches'; fig.Position = [1 1 2.1 2]; hold on;
b = bar(1:nClasses, m); %b.FaceColor = 'flat';

[ngroups, nbars] = size(m); x = nan(nbars, ngroups);
for i = 1:nbars, x(i,:) = b(i).XEndPoints; b(i).FaceColor = 'flat'; b(i).FaceColor = cmap(i,:); end
for i = 1:nbars, errorbar(x(i,:), m(:,i), zeros(size(m(:,i))), ci95(:,i), 'k', 'linestyle','none'); end

% b = bar(1:nClasses,m); b.FaceColor = 'flat';
% for i = 1:nClasses, b.CData(i,:) = colors(i,:); end
% errorbar(1:nClasses,m,zeros(size(m)),ci95,'k','LineStyle', 'none')

ylim([0 1]); xlabel('Class'); ylabel('Acc (%)'); hold off;
ax = gca; ax.TickLength = [0.06 0.06]; xlim([0.5 2.5])
yticks(0:0.2:1); yline(0.20, '--k')
xticks(0.9)


exportgraphics(fig,['figs/hvsm_acc' num2str(nFolds) '.jpg'], 'Resolution', 500)