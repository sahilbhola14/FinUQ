function plot_lins_discretization_error(true_discretization_error, discretization_max_error_bound, eval_num_intervals, lower_precision, save_folder)
    fg = figure();
    set(fg, 'Position', [100 1 1400 1200]);
    % [ha, pos] = tight_subplot(3,2,[0.13 .1],[.18 .05],[.075 .01]);
    [ha, pos] = tight_subplot(3,2,[0.1 .07],[.18 .05],[.1 .015]);
    for ii =1:length(eval_num_intervals)
        axes(ha(ii));
        [f_true, x_true] = ecdf(true_discretization_error(:, ii));
        [f_bound, x_bound] = ecdf(discretization_max_error_bound(:, ii));

        l0 = plot(x_true, f_true);
        l0.Color = 'k'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
        hold on;
        l1 = plot(x_bound, f_bound);
        l1.Color = 'k'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k'; l1.LineStyle = '--';

        hold off
        title('$M$ = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)
        grid on; box on;
        set(gca, 'XScale', 'log', 'xlim', [1e-6, 1e+2]);
        set(gca, 'YScale', 'log', 'ylim', [0, 1]);
        set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
    end

    global_xlabel = '$\xi = \Vert \hat{\tilde{\bf{u}}} - \tilde{\bf{u}}\Vert_{\infty}$';
    annotation('textbox', [0.43, 0.085, 0.2, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    global_ylabel = '$F_{\Xi}(\xi)$';
    annotation('textbox', [0.05, 0.4, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

    lg = legend('$\epsilon_{d;true}^{max}$', '$\epsilon_{d; model}$')
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 3);
    lg.Position = [0.38, 0.03, 0.3, 0.05];

end