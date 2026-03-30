clear; clc; close all;

curve_dir = 'alpha_post_curves';
files = dir(fullfile(curve_dir, '*_curve.csv'));

if isempty(files)
    error('No alpha curve csv files found.');
end

dataset_names = {};
info = struct([]);

for i = 1:length(files)
    fname = files(i).name;
    token = regexp(fname, '(.+)_alpha(-?1|0|1)_curve\.csv', 'tokens');

    if isempty(token)
        continue;
    end

    token = token{1};
    dataset = token{1};
    alpha_val = str2double(token{2});

    dataset_names{end+1} = dataset; %#ok<SAGROW>
    info(end+1).name = fname; %#ok<SAGROW>
    info(end).dataset = dataset;
    info(end).alpha = alpha_val;
end

dataset_names = unique(dataset_names, 'stable');
metric_name = 'test_acc1';

for d = 1:length(dataset_names)
    dataset = dataset_names{d};

    figure('Name', ['Alpha Ablation - ' dataset], 'Color', 'w');
    hold on; grid on;

    legend_entries = {};
    target_alphas = [-1, 0, 1];

    for a = 1:length(target_alphas)
        alpha_val = target_alphas(a);

        idx = find(strcmp({info.dataset}, dataset) & [info.alpha] == alpha_val, 1);
        if isempty(idx)
            continue;
        end

        T = readtable(fullfile(curve_dir, info(idx).name));
        plot(T.epoch, T.(metric_name), 'LineWidth', 2);

        legend_entries{end+1} = sprintf('alpha = %d', alpha_val); %#ok<SAGROW>
    end

    xlabel('Epoch');
    ylabel(strrep(metric_name, '_', '\_'));
    title(sprintf('%s: alpha ablation', dataset));
    legend(legend_entries, 'Location', 'best');
    set(gca, 'FontSize', 12);
end

% 最终结果柱状图
summary_file = fullfile('alpha_post_summary', 'alpha_ablation_summary.csv');
if exist(summary_file, 'file')
    S = readtable(summary_file);

    datasets = unique(S.dataset, 'stable');
    alphas = [-1, 0, 1];
    M = nan(length(datasets), length(alphas));

    for i = 1:length(datasets)
        for j = 1:length(alphas)
            idx = strcmp(S.dataset, datasets{i}) & S.alpha_post == alphas(j);
            if any(idx)
                M(i, j) = S.best_acc(find(idx, 1));
            end
        end
    end

    figure('Name', 'Alpha Ablation Summary', 'Color', 'w');
    bar(M, 'grouped');
    grid on;
    set(gca, 'XTick', 1:length(datasets), 'XTickLabel', datasets, 'FontSize', 12);
    xlabel('Dataset');
    ylabel('Best Acc@1');
    title('Alpha Ablation: Best Accuracy Comparison');
    legend({'alpha=-1', 'alpha=0', 'alpha=1'}, 'Location', 'best');
end