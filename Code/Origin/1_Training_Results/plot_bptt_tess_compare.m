clear; clc; close all;

curve_dir = 'bptt_tess_curves';
files = dir(fullfile(curve_dir, '*_curve.csv'));

if isempty(files)
    error('No BPTT/TESS curve csv files found.');
end

dataset_names = {};
info = struct([]);

for i = 1:length(files)
    fname = files(i).name;
    token = regexp(fname, '(.+)_(BPTT|TESS)_curve\.csv', 'tokens');

    if isempty(token)
        continue;
    end

    token = token{1};
    dataset = token{1};
    method = token{2};

    dataset_names{end+1} = dataset; %#ok<SAGROW>
    info(end+1).name = fname; %#ok<SAGROW>
    info(end).dataset = dataset;
    info(end).method = method;
end

dataset_names = unique(dataset_names, 'stable');
metric_name = 'test_acc1';

for d = 1:length(dataset_names)
    dataset = dataset_names{d};

    figure('Name', ['BPTT vs TESS - ' dataset], 'Color', 'w');
    hold on; grid on;

    legend_entries = {};
    methods = {'BPTT', 'TESS'};

    for m = 1:length(methods)
        method = methods{m};

        idx = find(strcmp({info.dataset}, dataset) & strcmp({info.method}, method), 1);
        if isempty(idx)
            continue;
        end

        T = readtable(fullfile(curve_dir, info(idx).name));
        plot(T.epoch, T.(metric_name), 'LineWidth', 2);

        legend_entries{end+1} = method; %#ok<SAGROW>
    end

    xlabel('Epoch');
    ylabel(strrep(metric_name, '_', '\_'));
    title(sprintf('%s: BPTT vs TESS', dataset));
    legend(legend_entries, 'Location', 'best');
    set(gca, 'FontSize', 12);
end

% 最终结果柱状图
summary_file = fullfile('bptt_tess_summary', 'bptt_tess_summary.csv');
if exist(summary_file, 'file')
    S = readtable(summary_file);

    datasets = unique(S.dataset, 'stable');
    methods = {'BPTT', 'TESS'};
    M = nan(length(datasets), length(methods));

    for i = 1:length(datasets)
        for j = 1:length(methods)
            idx = strcmp(S.dataset, datasets{i}) & strcmp(S.method, methods{j});
            if any(idx)
                M(i, j) = S.best_acc(find(idx, 1));
            end
        end
    end

    figure('Name', 'BPTT vs TESS Summary', 'Color', 'w');
    bar(M, 'grouped');
    grid on;
    set(gca, 'XTick', 1:length(datasets), 'XTickLabel', datasets, 'FontSize', 12);
    xlabel('Dataset');
    ylabel('Best Acc@1');
    title('BPTT vs TESS: Best Accuracy Comparison');
    legend({'BPTT', 'TESS'}, 'Location', 'best');
end