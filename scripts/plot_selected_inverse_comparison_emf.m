% Export selected WA inversion comparisons as publication-ready EMF files.
% Each ID uses its true resistivity model range as the shared color scale.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(rootDir, 'data');
runDir = fullfile(rootDir, 'liquid_network', 'runs');
outDir = fullfile(rootDir, 'inverse_comparison_emf');

ids = [4, 52, 126, 347, 434, 484, 487, 512];
lineLength = 122.5;
maxDepth = 18;

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

methods = {
    'wa_traditional', fullfile(dataDir, 'inv_input_wa', 'wainv_%d.mat'), ...
        'WA'
    'unet_wa_open_loop', fullfile(runDir, 'unet_wa_open_loop', ...
        'open_inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
    'unet_wa_closed_loop', fullfile(runDir, 'unet_wa_closed_loop', ...
        'closed_inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
    'lnn_wa_single_closed_loop', fullfile(runDir, ...
        'lnn_wa_single_closed_loop', 'evaluation', ...
        'inverse_predictions_mat', 'inverse_prediction_id_%d.mat'), ...
        'rho_pred'
    'lnn_multi_closed_loop', fullfile(runDir, 'lnn_multi_closed_loop', ...
        'inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
};

climById = zeros(numel(ids), 3);

for i = 1:numel(ids)
    id = ids(i);
    truthPath = fullfile(dataDir, 'rho', sprintf('rho_%d.mat', id));
    rhoTrue = load_matrix(truthPath, 'rho');
    assert(isequal(size(rhoTrue), [256, 1024]), ...
        'Unexpected truth size for ID %d: %s', id, mat2str(size(rhoTrue)));

    validTruth = rhoTrue(isfinite(rhoTrue));
    assert(~isempty(validTruth), 'Truth contains no finite values for ID %d.', id);
    climShared = [min(validTruth), max(validTruth)];
    assert(climShared(2) > climShared(1), ...
        'Truth color range is degenerate for ID %d.', id);
    climById(i, :) = [id, climShared];

    truthOutputPath = fullfile(outDir, ...
        sprintf('id_%d_rho_true.emf', id));
    plot_inverse_emf(rhoTrue, climShared, lineLength, ...
        maxDepth, truthOutputPath);
    fprintf('Saved: %s\n', truthOutputPath);

    for j = 1:size(methods, 1)
        methodName = methods{j, 1};
        inputPath = strrep(methods{j, 2}, '%d', num2str(id));
        valueKey = methods{j, 3};
        imageData = load_matrix(inputPath, valueKey);
        assert(isequal(size(imageData), size(rhoTrue)), ...
            'Size mismatch for ID %d, method %s: %s', ...
            id, methodName, mat2str(size(imageData)));

        outputPath = fullfile(outDir, ...
            sprintf('id_%d_%s.emf', id, methodName));
        plot_inverse_emf(imageData, climShared, lineLength, ...
            maxDepth, outputPath);
        fprintf('Saved: %s\n', outputPath);
    end
end

save(fullfile(outDir, 'rho_truth_clim_by_id.mat'), 'climById', 'ids');
fprintf('Completed: %d EMF files in %s\n', ...
    numel(ids) * (size(methods, 1) + 1), outDir);


function A = load_matrix(path, key)
    assert(exist(path, 'file') == 2, 'Missing file: %s', path);
    S = load(path);
    assert(isfield(S, key), 'Missing key "%s" in %s', key, path);
    A = double(S.(key));
    assert(isnumeric(A) && ismatrix(A), ...
        'Key "%s" is not a numeric matrix in %s', key, path);
    assert(all(isfinite(A(:))), 'Non-finite values in %s', path);
end


function plot_inverse_emf(A, climVals, lineLength, maxDepth, outPath)
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 1120, 680], ...
        'Renderer', 'painters', 'InvertHardcopy', 'off');
    ax = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.105, 0.180, 0.765, 0.720]);

    imagesc(ax, [0, lineLength], [0, maxDepth], A);
    set(ax, 'YDir', 'reverse');
    colormap(ax, jet(256));
    caxis(ax, climVals);

    set(ax, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38, ...
        'LineWidth', 1.8, 'Box', 'on', ...
        'TickDir', 'in', 'Layer', 'top', ...
        'XTick', [40, 80, 122.5], ...
        'XTickLabel', {'40', '80', '102'}, ...
        'YTick', [6, 12, 18], ...
        'YTickLabel', {'6', '12', '18'});
    xlabel(ax, 'Distance (m)', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    ylabel(ax, 'Depth (m)', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 38);
    xlim(ax, [0, lineLength]);
    ylim(ax, [0, maxDepth]);

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
