% Paper-style loss curves for the liquid project.
% Outputs PNG and EMF figures without top/right tick marks.
clear; clc; close all;

rootDir = fileparts(fileparts(mfilename('fullpath')));
runsDir = fullfile(rootDir, 'runs');
outDir = fullfile(runsDir, 'paper_loss_curves');
if ~exist(outDir, 'dir'), mkdir(outDir); end

experiments = struct();
experiments.sa_unet_ol.inverse = fullfile(runsDir, ...
    'unet_wa_open_loop', 'open_inverse', 'losses.csv');
experiments.sa_unet_cl.forward = fullfile(runsDir, ...
    'unet_wa_closed_loop', 'forward', 'losses.csv');
experiments.sa_unet_cl.inverse = fullfile(runsDir, ...
    'unet_wa_closed_loop', 'closed_inverse', 'losses.csv');
experiments.sa_lnn_unet_cl.forward = fullfile(runsDir, ...
    'lnn_wa_single_closed_loop', 'forward', 'losses.csv');
experiments.sa_lnn_unet_cl.inverse = fullfile(runsDir, ...
    'lnn_wa_single_closed_loop', 'inverse', 'losses.csv');
experiments.ma_lnn_unet_cl.forward = fullfile(runsDir, ...
    'lnn_multi_closed_loop', 'forward', 'losses.csv');
experiments.ma_lnn_unet_cl.inverse = fullfile(runsDir, ...
    'lnn_multi_closed_loop', 'inverse', 'losses.csv');

labels = struct( ...
    'sa_unet_ol', 'SA-U-Net-OL', ...
    'sa_unet_cl', 'SA-U-Net-CL', ...
    'sa_lnn_unet_cl', 'SA-PLB-U-Net-CL', ...
    'ma_lnn_unet_cl', 'MA-PLB-U-Net-CL');

colors = struct();
colors.sa_unet_ol = [0.45, 0.45, 0.45];
colors.sa_unet_cl = [0.85, 0.10, 0.10];
colors.sa_lnn_unet_cl = [0.00, 0.32, 0.85];
colors.ma_lnn_unet_cl = [0.05, 0.05, 0.05];

% Proposed method loss curves.
TpropF = read_loss_table(experiments.ma_lnn_unet_cl.forward);
plot_train_val_curve(TpropF.epoch, TpropF.train_total, TpropF.val_total, ...
    'Forward Loss Curve of MA-PLB-U-Net-CL', ...
    'Loss', fullfile(outDir, 'proposed_forward_loss_curve'));

TpropI = read_loss_table(experiments.ma_lnn_unet_cl.inverse);
plot_train_val_curve(TpropI.epoch, TpropI.train_total, TpropI.val_total, ...
    'Training and Validation Total Loss of MA-PLB-U-Net-CL', ...
    'Total Loss', fullfile(outDir, 'proposed_inverse_loss_curve'));

% Validation loss comparisons.
forwardCurves = {
    labels.sa_unet_cl, colors.sa_unet_cl, '--', ...
        read_curve(experiments.sa_unet_cl.forward, 'val_Lfwd_raw');
    labels.sa_lnn_unet_cl, colors.sa_lnn_unet_cl, '--', ...
        read_curve(experiments.sa_lnn_unet_cl.forward, 'val_Lfwd_raw');
    labels.ma_lnn_unet_cl, colors.ma_lnn_unet_cl, '-', ...
        read_curve(experiments.ma_lnn_unet_cl.forward, 'val_total');
};
plot_validation_comparison(forwardCurves, ...
    'Forward Validation Loss Curve Comparison', ...
    'Validation Loss', true, ...
    fullfile(outDir, 'forward_validation_loss_comparison_with_inset'));

inverseCurves = {
    labels.sa_unet_ol, colors.sa_unet_ol, '--', ...
        read_curve(experiments.sa_unet_ol.inverse, 'val_Linv_raw');
    labels.sa_unet_cl, colors.sa_unet_cl, '--', ...
        read_curve(experiments.sa_unet_cl.inverse, 'val_Linv_raw');
    labels.sa_lnn_unet_cl, colors.sa_lnn_unet_cl, '--', ...
        read_curve(experiments.sa_lnn_unet_cl.inverse, 'val_Linv_raw');
    labels.ma_lnn_unet_cl, colors.ma_lnn_unet_cl, '-', ...
        read_curve(experiments.ma_lnn_unet_cl.inverse, 'val_inv_raw');
};
plot_validation_comparison(inverseCurves, ...
    'Validation Inversion Loss Curves', ...
    'Validation L_{inv}', false, ...
    fullfile(outDir, 'inverse_validation_loss_comparison'));

fprintf('Saved paper loss curves to: %s\n', outDir);


function T = read_loss_table(path)
    assert(exist(path, 'file') == 2, 'Missing losses.csv: %s', path);
    opts = detectImportOptions(path, 'VariableNamingRule', 'preserve');
    T = readtable(path, opts);
end


function curve = read_curve(path, lossColumn)
    T = read_loss_table(path);
    assert(ismember('epoch', T.Properties.VariableNames), ...
        'Missing epoch column in %s', path);
    assert(ismember(lossColumn, T.Properties.VariableNames), ...
        'Missing %s column in %s', lossColumn, path);
    curve = struct();
    curve.epoch = T.epoch;
    curve.loss = T.(lossColumn);
end


function plot_train_val_curve(epoch, trainLoss, valLoss, ttl, ylab, basePath)
    fig = create_figure();
    ax = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.145, 0.150, 0.805, 0.760]);
    hold(ax, 'on');
    plot(ax, epoch, trainLoss, '-', 'LineWidth', 2.4, ...
        'Color', [0.00, 0.32, 0.85], 'DisplayName', 'Training Loss');
    plot(ax, epoch, valLoss, '-', 'LineWidth', 2.4, ...
        'Color', [0.85, 0.10, 0.10], 'DisplayName', 'Validation Loss');
    style_axes(ax, ttl, 'Epoch', ylab);
    legend(ax, 'Location', 'northeast', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 18, ...
        'FontWeight', 'bold');
    save_figure(fig, basePath);
end


function plot_validation_comparison(curves, ttl, ylab, addInset, basePath)
    fig = create_figure();
    ax = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.145, 0.150, 0.805, 0.760]);
    hold(ax, 'on');
    for i = 1:size(curves, 1)
        label = curves{i, 1};
        color = curves{i, 2};
        lineStyle = curves{i, 3};
        curve = curves{i, 4};
        plot(ax, curve.epoch, curve.loss, lineStyle, ...
            'LineWidth', line_width_for_style(lineStyle), ...
            'Color', color, 'DisplayName', label);
    end
    style_axes(ax, ttl, 'Epoch', ylab);
    legend(ax, 'Location', 'northeast', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 16, ...
        'FontWeight', 'bold');

    if addInset
        add_forward_inset(fig, curves);
    end
    save_figure(fig, basePath);
end


function add_forward_inset(fig, curves)
    maxEpoch = 0;
    for i = 1:size(curves, 1)
        maxEpoch = max(maxEpoch, max(curves{i, 4}.epoch));
    end
    xMax = maxEpoch;
    xMin = max(1, floor(maxEpoch * 0.75));

    yVals = [];
    for i = 1:size(curves, 1)
        curve = curves{i, 4};
        selected = curve.epoch >= xMin & curve.epoch <= xMax;
        yVals = [yVals; curve.loss(selected)]; %#ok<AGROW>
    end
    yVals = yVals(isfinite(yVals));
    yMin = min(yVals);
    yMax = max(yVals);
    yPad = max((yMax - yMin) * 0.18, eps);

    axInset = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [0.535, 0.335, 0.355, 0.315]);
    hold(axInset, 'on');
    for i = 1:size(curves, 1)
        label = curves{i, 1};
        color = curves{i, 2};
        lineStyle = curves{i, 3};
        curve = curves{i, 4};
        plot(axInset, curve.epoch, curve.loss, lineStyle, ...
            'LineWidth', max(1.6, line_width_for_style(lineStyle) - 0.3), ...
            'Color', color);
        if strcmp(label, 'MA-PLB-U-Net-CL')
            [minLoss, minIndex] = min(curve.loss);
            minEpoch = curve.epoch(minIndex);
            plot(axInset, minEpoch, minLoss, 'p', ...
                'MarkerSize', 13, 'MarkerFaceColor', [0.92, 0.00, 0.00], ...
                'MarkerEdgeColor', [0.92, 0.00, 0.00], ...
                'LineWidth', 1.2);
        end
    end
    xlim(axInset, [xMin, xMax]);
    ylim(axInset, [max(0, yMin - yPad), yMax + yPad]);
    set(axInset, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 14, ...
        'LineWidth', 1.2, 'TickDir', 'in', ...
        'Box', 'off', 'XMinorTick', 'off', 'YMinorTick', 'off');
    xlabel(axInset, 'Epoch', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 14);
    ylabel(axInset, 'Loss', 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 14);
end


function fig = create_figure()
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Units', 'pixels', 'Position', [100, 100, 980, 720], ...
        'Renderer', 'painters', 'InvertHardcopy', 'off');
end


function style_axes(ax, ttl, xlab, ylab)
    grid(ax, 'on');
    ax.GridAlpha = 0.16;
    ax.GridLineStyle = '-';
    set(ax, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 22, ...
        'LineWidth', 1.8, 'TickDir', 'in', ...
        'Box', 'off', 'XMinorTick', 'off', 'YMinorTick', 'off');
    xlabel(ax, xlab, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 24);
    ylabel(ax, ylab, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 24);
    title(ax, ttl, 'FontName', 'Times New Roman', ...
        'FontWeight', 'bold', 'FontSize', 24);
end


function w = line_width_for_style(lineStyle)
    if strcmp(lineStyle, '-')
        w = 2.8;
    else
        w = 2.2;
    end
end


function save_figure(fig, basePath)
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, [basePath '.png'], '-dpng', '-r600');
    print(fig, [basePath '.emf'], '-dmeta', '-r600');
    close(fig);
end
