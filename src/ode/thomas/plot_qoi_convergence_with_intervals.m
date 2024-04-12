function plot_qoi_convergence_with_intervals(qoi_analytical_mean, qoi_hp_mean, qoi_lp_mean, qoi_model_mean, qoi_true_sol, eval_num_intervals, higher_precision, lower_precision, eval_parameter_samples, save_folder)
    if higher_precision == 'double'
        higher_precision_label = 'fp64';
    elseif higher_precision == 'single'
        higher_precision_label = 'fp32';
    else
        error('Invalid higher precision')
    end
    if lower_precision == 'single'
        lower_precision_label = 'fp32';
    elseif lower_precision == 'half'
        lower_precision_label = 'fp16';
    else
        error('Invalid lower precision')
    end
    fg = figure();
    set(fg, 'Position', [100 100 1000 500]);
    [ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .09],[.1 .015]);
    axes(ha(1));
    l1 = plot(eval_num_intervals, qoi_true_sol - qoi_analytical_mean(end, :), 'DisplayName', 'Analytical');
    l1.Color = 'r'; l1.LineWidth = 2;
    hold on
    % l2 = plot(eval_num_intervals, qoi_true_sol - qoi_model_mean(end, :), 'DisplayName', 'Model');
    % l2.Color = 'k'; l2.LineWidth = 2;

    l3 = plot(eval_num_intervals, qoi_true_sol - qoi_hp_mean(end, :), 'DisplayName', higher_precision_label);
    l3.Color = 'b'; l3.LineWidth = 2; l3.LineStyle = '-'; l3.Marker = 's'; l3.MarkerSize = 15; l3.MarkerFaceColor = l3.Color; l3.MarkerEdgeColor = 'k';

    l4 = plot(eval_num_intervals, qoi_true_sol - qoi_lp_mean(end, :), 'DisplayName', lower_precision_label);
    l4.Color = 'g'; l4.LineWidth = 2; l4.LineStyle = '-.';
    hold off

    lg = legend();
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
    xticks([2.^(2:1:8)])
    xticklabels('$2^{' + string(2:1:8) + '}$')
    xlabel('$\#$ Intervals, $m$', 'Interpreter', 'latex')
    ylabel('$E[P] - \frac{1}{n}\sum_{i}\tilde{p}_i$', 'Interpreter', 'latex')
    % ylim([0, 1.5])
    title(['$n$ = ', num2str(eval_parameter_samples)], 'Interpreter', 'latex', 'FontSize', 20)
    set(gca, 'XScale', 'log')
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
    grid on; box on;
end