% Plot field-data inversion predictions with individual color ranges.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
caseId = getenv('LIQUID_FIELD_CASE');
if isempty(caseId), caseId = '1'; end
fieldDir = fullfile(rootDir, 'shice', caseId);
predictionDir = fullfile(fieldDir, 'results', 'predictions_mat');
traditionalDir = fullfile(fieldDir, 'processed', 'traditional_inv');
outDir = fullfile(fieldDir, 'results', 'emf');

if ~exist(outDir, 'dir'), mkdir(outDir); end

traditionalPath = fullfile(traditionalDir, 'wa_inv.mat');
geometry = load(traditionalPath, 'metadata');
assert(isfield(geometry, 'metadata'), ...
    'Missing geometry metadata in %s.', traditionalPath);
lineLength = max(double(geometry.metadata.output_x_range));
maxDepth = max(double(geometry.metadata.output_depth_range));
cropLeftMeters = 0;
useElevationAxis = false;
elevationRange = [81, 99];
if strcmp(caseId, '3')
    cropLeftMeters = 6;
    useElevationAxis = true;
end
plotLineLength = lineLength - cropLeftMeters;
assert(plotLineLength > 0, 'Invalid cropped line length for case %s.', caseId);
proposedPath = fullfile(predictionDir, ...
    'lnn_multi_closed_loop.mat');
traditionalName = sprintf('traditional_wn%s_modreslog', caseId);

items = {
    traditionalName, traditionalPath, 'WA'
    'unet_wa_open_loop', fullfile(predictionDir, ...
        'unet_wa_open_loop.mat'), 'rho_pred'
    'unet_wa_closed_loop', fullfile(predictionDir, ...
        'unet_wa_closed_loop.mat'), 'rho_pred'
    'lnn_wa_single_closed_loop', fullfile(predictionDir, ...
        'lnn_wa_single_closed_loop.mat'), 'rho_pred'
    'lnn_multi_closed_loop_proposed', proposedPath, 'rho_pred'
};

images = cell(size(items, 1), 1);
individualClim = zeros(size(items, 1), 2);
for k = 1:size(items, 1)
    images{k} = load_matrix(items{k, 2}, items{k, 3});
    if cropLeftMeters > 0
        images{k} = crop_left_by_distance(images{k}, lineLength, ...
            cropLeftMeters);
    end
    individualClim(k, :) = [min(images{k}(:)), max(images{k}(:))];
    assert(all(isfinite(individualClim(k, :))) && ...
        individualClim(k, 2) > individualClim(k, 1), ...
        'Invalid color range for %s.', items{k, 1});
end

for k = 1:size(items, 1)
    outputPath = fullfile(outDir, [items{k, 1}, '.emf']);
    plot_field_emf(images{k}, individualClim(k, :), plotLineLength, ...
        maxDepth, outputPath, useElevationAxis, elevationRange);
    fprintf('Saved: %s\n', outputPath);
end

methodNames = items(:, 1);
save(fullfile(outDir, 'individual_clim.mat'), ...
    'individualClim', 'methodNames', 'cropLeftMeters', 'plotLineLength');
fprintf('Completed: %d field-data EMF files in %s\n', ...
    size(items, 1), outDir);


function A = load_matrix(path, key)
    assert(exist(path, 'file') == 2, 'Missing file: %s', path);
    S = load(path);
    assert(isfield(S, key), 'Missing key "%s" in %s', key, path);
    A = double(S.(key));
    assert(isequal(size(A), [256, 1024]), ...
        'Unexpected matrix size in %s: %s', path, mat2str(size(A)));
    assert(all(isfinite(A(:))), 'Non-finite values in %s', path);
end


function B = crop_left_by_distance(A, lineLength, cropLeftMeters)
    width = size(A, 2);
    x = linspace(0, lineLength, width);
    keep = x >= cropLeftMeters;
    assert(any(keep), 'Cropping removed all columns.');
    B = A(:, keep);
end


function plot_field_emf(A, climVals, lineLength, maxDepth, outPath, ...
        useElevationAxis, elevationRange)
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 1120, 680], ...
        'Renderer', 'painters', 'InvertHardcopy', 'off');
    ax = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.105, 0.180, 0.765, 0.720]);

    if useElevationAxis
        imagesc(ax, [0, lineLength], elevationRange, flipud(A));
        set(ax, 'YDir', 'normal');
        yTickValues = [81, 85, 89, 93, 97, 99];
        yTickLabels = {'81', '85', '89', '93', '97', '99'};
        yLabelText = 'Elevation (m)';
        yLimits = elevationRange;
    else
        imagesc(ax, [0, lineLength], [0, maxDepth], A);
        set(ax, 'YDir', 'reverse');
        yTickValues = [4, 8, 16];
        yTickLabels = {'4', '8', '16'};
        yLabelText = 'Depth (m)';
        yLimits = [0, maxDepth];
    end
    colormap(ax, jet(256));
    caxis(ax, climVals);

    set(ax, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38, ...
        'LineWidth', 1.8, 'Box', 'on', ...
        'TickDir', 'in', 'Layer', 'top', ...
        'XTick', [40, 80, lineLength], ...
        'XTickLabel', arrayfun(@num2str, [40, 80, lineLength], ...
            'UniformOutput', false), ...
        'YTick', yTickValues, ...
        'YTickLabel', yTickLabels);
    xlabel(ax, 'Distance (m)', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    ylabel(ax, yLabelText, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    xlim(ax, [0, lineLength]);
    ylim(ax, yLimits);

    cb = colorbar(ax, 'Units', 'normalized', ...
        'Position', [0.895, 0.180, 0.032, 0.720]);
    set(cb, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38, ...
        'LineWidth', 1.5, 'TickDirection', 'in');
    title(cb, 'log_{10}(\rho)', 'Interpreter', 'tex', ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 32);

    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outPath, '-dmeta', '-r600');
    close(fig);
end
