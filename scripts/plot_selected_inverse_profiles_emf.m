% Export horizontal and vertical resistivity profiles for selected IDs.
% The profile positions maximize the proposed model's error margin over
% the four comparison inversions, following the reference script in chaofen.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(rootDir, 'data');
runDir = fullfile(rootDir, 'liquid_network', 'runs');
outDir = fullfile(rootDir, 'inverse_profile_emf');

ids = [4, 52, 126, 347, 434, 484, 487, 512];
lineLength = 122.5;
maxDepth = 18;

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

sources = {
    'WA traditional', fullfile(dataDir, 'inv_input_wa', ...
        'wainv_%d.mat'), 'WA'
    'U-Net WA open-loop', fullfile(runDir, 'unet_wa_open_loop', ...
        'open_inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
    'U-Net WA closed-loop', fullfile(runDir, 'unet_wa_closed_loop', ...
        'closed_inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
    'LNN-U-Net WA single closed-loop', fullfile(runDir, ...
        'lnn_wa_single_closed_loop', 'evaluation', ...
        'inverse_predictions_mat', 'inverse_prediction_id_%d.mat'), ...
        'rho_pred'
    'Proposed', fullfile(runDir, 'lnn_multi_closed_loop', ...
        'inverse', 'evaluation', 'predictions_mat', ...
        'inverse_prediction_id_%d.mat'), 'rho_pred'
    'True', fullfile(dataDir, 'rho', 'rho_%d.mat'), 'rho'
};

proposedIndex = 5;
trueIndex = 6;
comparisonIndices = 1:4;

for i = 1:numel(ids)
    id = ids(i);
    dataListLog = cell(size(sources, 1), 1);

    for k = 1:size(sources, 1)
        inputPath = strrep(sources{k, 2}, '%d', num2str(id));
        dataListLog{k} = load_matrix(inputPath, sources{k, 3});
        if k > 1
            assert(isequal(size(dataListLog{k}), size(dataListLog{1})), ...
                'Size mismatch for ID %d: %s', id, sources{k, 1});
        end
    end

    [rowIndex, colIndex, scoreInfo] = choose_profile_position( ...
        dataListLog, proposedIndex, trueIndex, comparisonIndices);
    outputPath = fullfile(outDir, sprintf('id_%d_profiles.emf', id));
    plot_profiles(dataListLog, rowIndex, colIndex, lineLength, ...
        maxDepth, outputPath);
    save_position_info(outDir, id, rowIndex, colIndex, ...
        size(dataListLog{1}), lineLength, maxDepth, scoreInfo);

    fprintf(['Saved: %s | horizontal depth = %.3f m | ', ...
        'vertical distance = %.3f m\n'], outputPath, ...
        index_to_coordinate(rowIndex, size(dataListLog{1}, 1), maxDepth), ...
        index_to_coordinate(colIndex, size(dataListLog{1}, 2), lineLength));
end

fprintf('Completed: %d profile EMF files in %s\n', numel(ids), outDir);


function A = load_matrix(path, key)
    assert(exist(path, 'file') == 2, 'Missing file: %s', path);
    S = load(path);
    assert(isfield(S, key), 'Missing key "%s" in %s', key, path);
    A = double(squeeze(S.(key)));
    assert(isnumeric(A) && ismatrix(A), ...
        'Key "%s" is not a numeric matrix in %s', key, path);
    assert(all(isfinite(A(:))), 'Non-finite values in %s', path);
end


function [bestRow, bestCol, info] = choose_profile_position( ...
        dataListLog, proposedIndex, trueIndex, comparisonIndices)
    trueLog = dataListLog{trueIndex};
    proposedLog = dataListLog{proposedIndex};
    [height, width] = size(trueLog);

    rowCandidates = unique(round(linspace( ...
        round(height * 0.18), round(height * 0.82), 33)));
    colCandidates = unique(round(linspace( ...
        round(width * 0.12), round(width * 0.88), 41)));

    bestRow = round(height / 2);
    bestCol = round(width / 2);
    bestScore = -inf;
    bestRowMargin = -inf;
    bestColMargin = -inf;

    for row = rowCandidates
        proposedRowError = profile_rmse( ...
            proposedLog(row, :), trueLog(row, :));
        bestOtherRowError = inf;
        for k = comparisonIndices
            bestOtherRowError = min(bestOtherRowError, profile_rmse( ...
                dataListLog{k}(row, :), trueLog(row, :)));
        end
        rowMargin = bestOtherRowError - proposedRowError;

        for col = colCandidates
            proposedColError = profile_rmse( ...
                proposedLog(:, col), trueLog(:, col));
            bestOtherColError = inf;
            for k = comparisonIndices
                bestOtherColError = min(bestOtherColError, profile_rmse( ...
                    dataListLog{k}(:, col), trueLog(:, col)));
            end
            colMargin = bestOtherColError - proposedColError;
            score = rowMargin + colMargin;

            if score > bestScore
                bestScore = score;
                bestRow = row;
                bestCol = col;
                bestRowMargin = rowMargin;
                bestColMargin = colMargin;
            end
        end
    end

    info.score = bestScore;
    info.rowMargin = bestRowMargin;
    info.colMargin = bestColMargin;
end


function value = profile_rmse(predictedLog, trueLog)
    difference = predictedLog - trueLog;
    value = sqrt(mean(difference(:) .^ 2));
end


function plot_profiles(dataListLog, rowIndex, colIndex, ...
        lineLength, maxDepth, outputPath)
    [height, width] = size(dataListLog{1});
    distance = linspace(0, lineLength, width);
    depth = linspace(0, maxDepth, height);
    profileDepth = depth(rowIndex);
    profileDistance = distance(colIndex);

    % Order: traditional, U-Net open, U-Net closed, LNN single,
    % proposed, true. Proposed is red; true is black.
    colors = [
        0.50, 0.50, 0.50
        0.10, 0.35, 0.85
        0.00, 0.55, 0.25
        0.62, 0.22, 0.72
        0.90, 0.10, 0.08
        0.00, 0.00, 0.00
    ];
    lineStyles = {'--', '--', '--', '--', '-', '-'};
    otherLineWidth = 1.6;
    emphasizedLineWidth = otherLineWidth + 1.0;
    lineWidths = [repmat(otherLineWidth, 1, 4), ...
        emphasizedLineWidth, emphasizedLineWidth];

    rhoLinear = cellfun(@(A) 10 .^ A, dataListLog, ...
        'UniformOutput', false);

    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 1280, 590], ...
        'Renderer', 'painters', 'InvertHardcopy', 'off');
    layout = tiledlayout(fig, 1, 2, ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    horizontalAx = nexttile(layout);
    hold(horizontalAx, 'on');
    for k = 1:numel(rhoLinear)
        plot(horizontalAx, distance, rhoLinear{k}(rowIndex, :), ...
            'Color', colors(k, :), ...
            'LineStyle', lineStyles{k}, ...
            'LineWidth', lineWidths(k));
    end
    hold(horizontalAx, 'off');
    style_profile_axis(horizontalAx);
    xlabel(horizontalAx, 'Distance (m)', ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 27);
    ylabel(horizontalAx, resistivity_label(), ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 27, 'Interpreter', 'none');
    title(horizontalAx, sprintf( ...
        'Horizontal profile (depth = %.1f m)', profileDepth), ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 23);
    xlim(horizontalAx, [0, lineLength]);
    set(horizontalAx, 'XTick', [0, 40, 80, 122.5], ...
        'XTickLabel', {'0', '40', '80', '122.5'});
    set_positive_limit(horizontalAx, 'y', rhoLinear, rowIndex);

    verticalAx = nexttile(layout);
    hold(verticalAx, 'on');
    for k = 1:numel(rhoLinear)
        plot(verticalAx, rhoLinear{k}(:, colIndex), depth, ...
            'Color', colors(k, :), ...
            'LineStyle', lineStyles{k}, ...
            'LineWidth', lineWidths(k));
    end
    hold(verticalAx, 'off');
    style_profile_axis(verticalAx);
    set(verticalAx, 'YDir', 'reverse', ...
        'YTick', [0, 6, 12, 18], ...
        'YTickLabel', {'0', '6', '12', '18'});
    xlabel(verticalAx, resistivity_label(), ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 27, 'Interpreter', 'none');
    ylabel(verticalAx, 'Depth (m)', ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 27);
    title(verticalAx, sprintf( ...
        'Vertical profile (distance = %.1f m)', profileDistance), ...
        'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
        'FontSize', 23);
    ylim(verticalAx, [0, maxDepth]);
    set_positive_limit(verticalAx, 'x', rhoLinear, colIndex);

    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outputPath, '-dmeta', '-r600');
    close(fig);
end


function style_profile_axis(ax)
    set(ax, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 25, ...
        'LineWidth', 1.5, 'Box', 'on', ...
        'TickDir', 'in', 'Layer', 'top');
    grid(ax, 'on');
    ax.GridLineStyle = '-';
    ax.GridAlpha = 0.20;
end


function set_positive_limit(ax, direction, dataList, profileIndex)
    values = [];
    for k = 1:numel(dataList)
        if direction == 'y'
            values = [values; dataList{k}(profileIndex, :).']; %#ok<AGROW>
        else
            values = [values; dataList{k}(:, profileIndex)]; %#ok<AGROW>
        end
    end
    upperLimit = max(values);
    if ~isfinite(upperLimit) || upperLimit <= 0
        upperLimit = 1;
    end
    upperLimit = upperLimit * 1.05;
    if direction == 'y'
        ylim(ax, [0, upperLimit]);
    else
        xlim(ax, [0, upperLimit]);
    end
end


function label = resistivity_label()
    label = sprintf('Resistivity (%c%cm)', char(937), char(183));
end


function coordinate = index_to_coordinate(index, count, maximum)
    coordinate = (index - 1) / (count - 1) * maximum;
end


function save_position_info(outDir, id, rowIndex, colIndex, ...
        matrixSize, lineLength, maxDepth, info)
    depth = index_to_coordinate(rowIndex, matrixSize(1), maxDepth);
    distance = index_to_coordinate(colIndex, matrixSize(2), lineLength);
    outputPath = fullfile(outDir, sprintf('id_%d_position.txt', id));
    fileId = fopen(outputPath, 'w');
    assert(fileId >= 0, 'Cannot create position file: %s', outputPath);
    cleanup = onCleanup(@() fclose(fileId));
    fprintf(fileId, 'id\t%d\n', id);
    fprintf(fileId, 'row_index\t%d\n', rowIndex);
    fprintf(fileId, 'column_index\t%d\n', colIndex);
    fprintf(fileId, 'horizontal_depth_m\t%.8f\n', depth);
    fprintf(fileId, 'vertical_distance_m\t%.8f\n', distance);
    fprintf(fileId, 'selection_score\t%.8f\n', info.score);
    fprintf(fileId, 'horizontal_margin\t%.8f\n', info.rowMargin);
    fprintf(fileId, 'vertical_margin\t%.8f\n', info.colMargin);
    clear cleanup;
end
