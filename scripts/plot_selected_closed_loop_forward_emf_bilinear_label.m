% Plot-only experiment: rebuild WA forward labels from raw scatter data with
% bilinear vertical resizing. Original MAT labels and trained models remain unchanged.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(rootDir, 'data');
runDir = fullfile(rootDir, 'liquid_network', 'runs');
outDir = fullfile(rootDir, ...
    'closed_loop_forward_comparison_emf_bilinear_label_plot_only');
matOutDir = fullfile(outDir, 'display_mat');
previewDir = fullfile(outDir, 'preview_png');

ids = [4, 52, 126, 347, 434, 484, 487, 512];
lineLength = 122.5;
targetHeight = 256;
targetWidth = 1024;
filterSize = 5;
xGrid = linspace(-lineLength / 2, lineLength / 2, targetWidth);

if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(matOutDir, 'dir'), mkdir(matOutDir); end
if ~exist(previewDir, 'dir'), mkdir(previewDir); end

predictionMethods = {
    'unet_wa_closed_loop', fullfile(runDir, 'unet_wa_closed_loop', ...
        'forward', 'evaluation', 'predictions_mat', ...
        'forward_prediction_id_%d.mat'), 'WA_pred', 'mask'
    'lnn_wa_single_closed_loop', fullfile(runDir, ...
        'lnn_wa_single_closed_loop', 'evaluation', ...
        'forward_predictions_mat', 'forward_prediction_id_%d.mat'), ...
        'WA_pred', 'mask'
    'lnn_multi_closed_loop', fullfile(runDir, 'lnn_multi_closed_loop', ...
        'forward', 'evaluation', 'wa', 'predictions_mat', ...
        'forward_prediction_id_%d.mat'), 'WA_pred', 'mask'
};

climById = zeros(numel(ids), 3);
maxLevelById = zeros(numel(ids), 2);
differenceStats = zeros(numel(ids), 4);

for i = 1:numel(ids)
    id = ids(i);
    originalPath = fullfile(dataDir, 'wa_256_layered', ...
        sprintf('rhoa_2d_%d.mat', id));
    [nearestLabel, originalMask] = load_forward_matrix( ...
        originalPath, 'img', 'mask');
    originalMask = originalMask > 0;

    rawPath = fullfile(dataDir, 'wa', ...
        sprintf('raw_scatter_%d.mat', id));
    [bilinearLabel, bilinearMask, maxLevel] = ...
        rebuild_bilinear_label(rawPath, xGrid, targetHeight, targetWidth);
    assert(isequal(bilinearMask, originalMask), ...
        'Rebuilt mask differs from the original mask for ID %d.', id);

    nearestFiltered = masked_mean_filter( ...
        nearestLabel, originalMask, filterSize);
    bilinearFiltered = masked_mean_filter( ...
        bilinearLabel, originalMask, filterSize);
    validTruth = bilinearFiltered(originalMask & isfinite(bilinearFiltered));
    assert(~isempty(validTruth), ...
        'Bilinear label contains no valid values for ID %d.', id);
    climShared = [min(validTruth), max(validTruth)];
    assert(climShared(2) > climShared(1), ...
        'Bilinear truth color range is degenerate for ID %d.', id);
    climById(i, :) = [id, climShared];
    maxLevelById(i, :) = [id, maxLevel];

    delta = bilinearFiltered(originalMask) - nearestFiltered(originalMask);
    differenceStats(i, :) = [id, mean(abs(delta)), ...
        sqrt(mean(delta .^ 2)), max(abs(delta))];

    save_and_plot('wa_true_nearest', nearestFiltered, originalMask, ...
        id, climShared, lineLength, maxLevel, filterSize, ...
        'nearest label + masked 5x5 mean filter', outDir, matOutDir);
    save_and_plot('wa_true_bilinear', bilinearFiltered, originalMask, ...
        id, climShared, lineLength, maxLevel, filterSize, ...
        'bilinear label + masked 5x5 mean filter', outDir, matOutDir);

    for j = 1:size(predictionMethods, 1)
        methodName = predictionMethods{j, 1};
        inputPath = strrep(predictionMethods{j, 2}, '%d', num2str(id));
        [prediction, predictionMask] = load_forward_matrix( ...
            inputPath, predictionMethods{j, 3}, predictionMethods{j, 4});
        assert(isequal(size(prediction), size(nearestLabel)), ...
            'Size mismatch for ID %d, method %s.', id, methodName);
        assert(isequal(predictionMask > 0, originalMask), ...
            'Mask mismatch for ID %d, method %s.', id, methodName);
        predictionFiltered = masked_mean_filter( ...
            prediction, originalMask, filterSize);
        save_and_plot(methodName, predictionFiltered, originalMask, ...
            id, climShared, lineLength, maxLevel, filterSize, ...
            'unchanged prediction + masked 5x5 mean filter', ...
            outDir, matOutDir);
    end

    previewPath = fullfile(previewDir, ...
        sprintf('id_%d_nearest_vs_bilinear.png', id));
    plot_label_preview(nearestFiltered, bilinearFiltered, ...
        originalMask, climShared, lineLength, maxLevel, id, previewPath);
end

save(fullfile(outDir, 'bilinear_label_plot_metadata.mat'), ...
    'climById', 'maxLevelById', 'differenceStats', 'ids', ...
    'filterSize', 'lineLength');

fid = fopen(fullfile(outDir, 'README.txt'), 'w');
assert(fid >= 0, 'Cannot create README.txt in %s.', outDir);
fprintf(fid, ['Plot-only comparison. Original training labels, prediction MAT files, ' ...
    'and model weights were not modified.\n']);
fprintf(fid, ['WA labels were rebuilt from raw_scatter data and resized vertically ' ...
    'with bilinear interpolation.\n']);
fprintf(fid, ['Masks use nearest-neighbor resizing. The existing masked 5x5 mean ' ...
    'filter is retained for both labels and predictions.\n']);
fprintf(fid, 'differenceStats columns: ID, MAE, RMSE, maximum absolute difference.\n');
fclose(fid);

fprintf('Completed: %d EMF files and %d PNG previews in %s\n', ...
    numel(ids) * (size(predictionMethods, 1) + 2), numel(ids), outDir);


function [A, mask] = load_forward_matrix(path, valueKey, maskKey)
    assert(exist(path, 'file') == 2, 'Missing file: %s', path);
    S = load(path);
    assert(isfield(S, valueKey), ...
        'Missing key "%s" in %s', valueKey, path);
    assert(isfield(S, maskKey), ...
        'Missing key "%s" in %s', maskKey, path);
    A = double(S.(valueKey));
    mask = double(S.(maskKey));
    assert(isnumeric(A) && ismatrix(A), ...
        'Key "%s" is not a numeric matrix in %s', valueKey, path);
    assert(isequal(size(A), size(mask)), ...
        'Value/mask size mismatch in %s', path);
    assert(all(isfinite(A(:))), 'Non-finite values in %s', path);
end


function [label, mask, maxLevel] = rebuild_bilinear_label( ...
        rawPath, xGrid, targetHeight, targetWidth)
    assert(exist(rawPath, 'file') == 2, 'Missing file: %s', rawPath);
    S = load(rawPath, 'x_mid', 'levels', 'rhoa');
    required = {'x_mid', 'levels', 'rhoa'};
    for k = 1:numel(required)
        assert(isfield(S, required{k}), ...
            'Missing key "%s" in %s.', required{k}, rawPath);
    end
    x = double(S.x_mid(:));
    levels = round(double(S.levels(:)));
    rhoa = double(S.rhoa(:));
    valid = isfinite(x) & isfinite(levels) & isfinite(rhoa) & rhoa > 0;
    x = x(valid);
    levels = levels(valid);
    values = log10(rhoa(valid));
    maxLevel = max(levels);
    assert(maxLevel >= 2 && all(ismember(1:maxLevel, unique(levels)')), ...
        'Invalid or incomplete levels in %s.', rawPath);

    strips = nan(maxLevel, targetWidth);
    for level = 1:maxLevel
        selected = levels == level;
        xLevel = x(selected);
        valueLevel = values(selected);
        [xLevel, order] = sort(xLevel);
        valueLevel = valueLevel(order);
        [xLevel, uniqueIndex] = unique(xLevel, 'stable');
        valueLevel = valueLevel(uniqueIndex);
        if numel(xLevel) == 1
            [~, column] = min(abs(xGrid - xLevel));
            strips(level, column) = valueLevel;
        else
            strips(level, :) = interp1( ...
                xLevel, valueLevel, xGrid, 'linear', NaN);
        end
    end

    stripMask = isfinite(strips);
    stripValues = strips;
    stripValues(~stripMask) = 0;
    weightedValues = imresize( ...
        stripValues, [targetHeight, targetWidth], 'bilinear');
    weights = imresize( ...
        double(stripMask), [targetHeight, targetWidth], 'bilinear');
    mask = imresize( ...
        double(stripMask), [targetHeight, targetWidth], 'nearest') > 0.5;
    label = weightedValues ./ max(weights, eps);
    label(~mask | weights <= eps) = NaN;
    assert(all(isfinite(label(mask))), ...
        'Bilinear interpolation produced invalid values in %s.', rawPath);
end


function filtered = masked_mean_filter(A, mask, filterSize)
    kernel = ones(filterSize, filterSize);
    valid = mask & isfinite(A);
    values = A;
    values(~valid) = 0;
    weightedSum = conv2(values, kernel, 'same');
    validCount = conv2(double(valid), kernel, 'same');
    filtered = weightedSum ./ max(validCount, 1);
    filtered(~mask | validCount == 0) = NaN;
end


function save_and_plot(methodName, imageDisplayed, mask, id, ...
        climShared, lineLength, maxLevel, filterSize, processing, ...
        outDir, matOutDir)
    outputPath = fullfile(outDir, ...
        sprintf('id_%d_%s.emf', id, methodName));
    plot_forward_emf(imageDisplayed, climShared, ...
        lineLength, maxLevel, outputPath);
    matPath = fullfile(matOutDir, ...
        sprintf('id_%d_%s_display.mat', id, methodName));
    save(matPath, 'imageDisplayed', 'mask', 'climShared', ...
        'maxLevel', 'filterSize', 'processing');
    fprintf('Saved: %s\n', outputPath);
end


function plot_forward_emf(A, climVals, lineLength, maxLevel, outPath)
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 1120, 680], ...
        'Renderer', 'painters', 'InvertHardcopy', 'off');
    ax = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.105, 0.180, 0.765, 0.720]);
    imageHandle = imagesc(ax, [0, lineLength], [1, maxLevel], A);
    imageHandle.AlphaData = double(isfinite(A));
    ax.Color = [1, 1, 1];
    set(ax, 'YDir', 'reverse');
    colormap(ax, jet(256));
    caxis(ax, climVals);

    levelTicks = unique(round([maxLevel / 4, maxLevel / 2, ...
        3 * maxLevel / 4, maxLevel]));
    levelTicks(levelTicks < 1) = [];
    levelLabels = arrayfun(@num2str, levelTicks, ...
        'UniformOutput', false);
    set(ax, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38, ...
        'LineWidth', 1.8, 'Box', 'on', ...
        'TickDir', 'in', 'Layer', 'top', ...
        'XTick', [40, 80, 122.5], ...
        'XTickLabel', {'40', '80', '102'}, ...
        'YTick', levelTicks, 'YTickLabel', levelLabels);
    xlabel(ax, 'Distance (m)', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    ylabel(ax, 'Level', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    xlim(ax, [0, lineLength]);
    ylim(ax, [1, maxLevel]);
    cb = colorbar(ax, 'Units', 'normalized', ...
        'Position', [0.895, 0.180, 0.032, 0.720]);
    set(cb, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38, ...
        'LineWidth', 1.5, 'TickDirection', 'in');
    title(cb, 'log_{10}(\rho_a)', 'Interpreter', 'tex', ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 32);
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outPath, '-dmeta', '-r600');
    close(fig);
end


function plot_label_preview(nearestLabel, bilinearLabel, mask, ...
        climVals, lineLength, maxLevel, id, outPath)
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 1600, 620], ...
        'Renderer', 'opengl');
    tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    labels = {nearestLabel, bilinearLabel};
    titles = {'Original nearest-neighbor label', 'Plot-only bilinear label'};
    for k = 1:2
        ax = nexttile;
        handle = imagesc(ax, [0, lineLength], [1, maxLevel], labels{k});
        handle.AlphaData = double(mask & isfinite(labels{k}));
        ax.Color = [1, 1, 1];
        set(ax, 'YDir', 'reverse', 'FontName', 'Times New Roman', ...
            'FontWeight', 'bold', 'FontSize', 20, 'LineWidth', 1.3, ...
            'XTick', [40, 80, 122.5], ...
            'XTickLabel', {'40', '80', '102'}, ...
            'YTick', [4, 8, 12, 16]);
        colormap(ax, jet(256));
        caxis(ax, climVals);
        xlim(ax, [0, lineLength]);
        ylim(ax, [1, maxLevel]);
        xlabel(ax, 'Distance (m)');
        ylabel(ax, 'Level');
        title(ax, titles{k});
        colorbar(ax);
    end
    sgtitle(sprintf('WA forward label ID %d', id), ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 22);
    print(fig, outPath, '-dpng', '-r220');
    close(fig);
end
