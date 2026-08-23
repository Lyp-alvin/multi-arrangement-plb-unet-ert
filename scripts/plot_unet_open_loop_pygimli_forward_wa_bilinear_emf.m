% Forward unet_wa_open_loop inverse predictions with pyGIMLi, then plot WA
% forward responses using plot-only bilinear interpolation. Color scale is
% shared with wa_true_bilinear for each ID.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(rootDir, 'data');
forwardDir = fullfile(rootDir, 'unet_open_loop_pygimli_forward_wa', 'wa');
outDir = fullfile(rootDir, ...
    'unet_open_loop_pygimli_forward_wa_bilinear_emf');
matOutDir = fullfile(outDir, 'display_mat');

ids = [4, 52, 126, 347, 434, 484, 487, 512];
lineLength = 122.5;
targetHeight = 256;
targetWidth = 1024;
filterSize = 0;
xGrid = linspace(-lineLength / 2, lineLength / 2, targetWidth);

if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(matOutDir, 'dir'), mkdir(matOutDir); end

climById = zeros(numel(ids), 3);
maxLevelById = zeros(numel(ids), 2);

for i = 1:numel(ids)
    id = ids(i);
    originalPath = fullfile(dataDir, 'wa_256_layered', ...
        sprintf('rhoa_2d_%d.mat', id));
    [~, originalMask] = load_forward_matrix(originalPath, 'img', 'mask');
    originalMask = originalMask > 0;

    truthRawPath = fullfile(dataDir, 'wa', sprintf('raw_scatter_%d.mat', id));
    [truthBilinear, truthMask, maxLevel] = rebuild_bilinear_label( ...
        truthRawPath, xGrid, targetHeight, targetWidth);
    assert(isequal(truthMask, originalMask), ...
        'Truth bilinear mask differs from original mask for ID %d.', id);

    predRawPath = fullfile(forwardDir, ...
        sprintf('raw_scatter_unet_open_loop_%d.mat', id));
    [predBilinear, predMask, predMaxLevel] = rebuild_bilinear_label( ...
        predRawPath, xGrid, targetHeight, targetWidth);
    assert(maxLevel == predMaxLevel, ...
        'Max level mismatch for ID %d: truth=%d prediction=%d.', ...
        id, maxLevel, predMaxLevel);
    assert(isequal(predMask, originalMask), ...
        'Prediction mask differs from original mask for ID %d.', id);

    truthDisplayed = truthBilinear;
    predDisplayed = predBilinear;
    validTruth = truthDisplayed(originalMask & isfinite(truthDisplayed));
    assert(~isempty(validTruth), 'No finite bilinear truth values for ID %d.', id);
    climShared = [min(validTruth), max(validTruth)];
    climById(i, :) = [id, climShared];
    maxLevelById(i, :) = [id, maxLevel];

    save_and_plot('wa_true_bilinear', truthDisplayed, originalMask, ...
        id, climShared, lineLength, maxLevel, filterSize, ...
        'truth raw_scatter + bilinear vertical resize, no mean filter', ...
        outDir, matOutDir);
    save_and_plot('unet_wa_open_loop_pygimli_forward', predDisplayed, originalMask, ...
        id, climShared, lineLength, maxLevel, filterSize, ...
        'unet open-loop inverse prediction -> pyGIMLi WA forward -> bilinear vertical resize, no mean filter', ...
        outDir, matOutDir);
end

save(fullfile(outDir, 'unet_open_loop_pygimli_forward_plot_metadata.mat'), ...
    'ids', 'climById', 'maxLevelById', 'filterSize', 'lineLength');

fid = fopen(fullfile(outDir, 'README.txt'), 'w');
assert(fid >= 0, 'Cannot create README.txt in %s.', outDir);
fprintf(fid, ['unet_wa_open_loop inverse predictions were physically forward-modeled ' ...
    'with pyGIMLi using WA, 50 electrodes, and 2.5 m spacing.\n']);
fprintf(fid, ['Forward scatter data were rebuilt with bilinear vertical resizing, ' ...
    'then plotted directly without mean filtering.\n']);
fprintf(fid, ['Each ID uses the unfiltered wa_true_bilinear valid range as the shared ' ...
    'colorbar for truth and prediction.\n']);
fclose(fid);

fprintf('Completed: %d EMF files in %s\n', numel(ids) * 2, outDir);


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
        'XTickLabel', {'40', '80', '122.5'}, ...
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
