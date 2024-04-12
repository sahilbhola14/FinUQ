function plot_qoi_abs_error_cdf(qoi_abs_error_true, qoi_abs_error_model, qoi_deterministic_e_fwd_bound, qoi_higham_e_fwd_bound, ...
                       qoi_bernstein_e_fwd_bound, eval_num_intervals, lower_precision, ...
                       prob_bound_confidence_levels, save_folder)
    assert (length(prob_bound_confidence_levels) == 1, 'Only one confidence level is supported for now (Update plotting legend for multiple support)');
    fg = figure();
    set(fg, 'Position', [100 1 1400 1200]);
    % [ha, pos] = tight_subplot(3,2,[0.13 .1],[.18 .05],[.075 .01]);
    [ha, pos] = tight_subplot(3,2,[0.1 .07],[.18 .05],[.1 .015]);
    set_line_width = 2.75;
    for ii =1:length(eval_num_intervals)
        axes(ha(ii));
        [f_true, x_true] = ecdf(qoi_abs_error_true(:, ii));
        [f_model, x_model] = ecdf(qoi_abs_error_model(:, ii));
        [f_deter, x_deter] = ecdf(qoi_deterministic_e_fwd_bound(:, ii));
        % [f_discretization, x_discretization] = ecdf(discretization_max_error(:, ii));
        l0 = plot(x_true, f_true, 'DisplayName', '$\epsilon_{fwd}^{true}$');
        l0.Color = 'k'; l0.LineWidth = set_line_width; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
        hold on;

        l1 = plot(x_model, f_model, 'DisplayName', '$\epsilon_{fwd}^{model}$');
        l1.Color = 'r'; l1.LineWidth = set_line_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';

        l2 = plot(x_deter, f_deter, 'DisplayName', '$\tau_{ODE}$');
        l2.Color = 'b'; l2.LineWidth = set_line_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 'none'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';

        for jj = 1:length(prob_bound_confidence_levels)
            [f_higham, x_higham] = ecdf(qoi_higham_e_fwd_bound(:, ii, jj));
            [f_bernstein, x_bernstein] = ecdf(qoi_bernstein_e_fwd_bound(:, ii, jj));

            l3 = plot(x_higham, f_higham, 'DisplayName', ['$\hat{\tau}_{ODE}(\lambda_{ODE}^h(\alpha = ', num2str(prob_bound_confidence_levels(jj)), '))$']);
            l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = set_line_width; l3.MarkerFaceColor = l3.Color; l3.Marker = 'none'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';


            l4 = plot(x_bernstein, f_bernstein, 'DisplayName', ['$\hat{\tau}_{ODE}(\lambda_{ODE}^b(\alpha = ', num2str(prob_bound_confidence_levels(jj)), '))$']);
            l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = set_line_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'none'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

        end

        % l4 = plot(x_discretization, f_discretization, 'DisplayName', 'Discretization error');
        % l4.Color = [0.9290, 0.6940, 0.1250]; l4.LineStyle = '-.'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 'none'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

        ylim([0, 1]);
        xlim([1e-10, 1e+0])

        hold off;
        % title('Number of intervals = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)
        title('$M$ = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)
        % xlabel('$\xi$', 'Interpreter', 'latex', 'FontSize', 20)
        % ylabel('$F_{\Xi}(\xi)$', 'Interpreter', 'latex', 'FontSize', 20)

        % xlabel('$\xi = |\hat{\tilde{p}} - \tilde{p}|$', 'Interpreter', 'latex', 'FontSize', 20)
        % ylabel('$F_{\Xi}(\xi)$', 'Interpreter', 'latex', 'FontSize', 20)

        grid on; box on;

        set(gca, 'XScale', 'log');
        set(gca, 'YScale', 'log');
        set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.5);
        % lg  = legend();
        % set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southeast', 'NumColumns', 2);

    end

    global_xlabel = '$\xi = \vert \hat{\tilde{p}} - \tilde{p}\vert$';
    annotation('textbox', [0.43, 0.085, 0.2, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    global_ylabel = '$F_{\Xi}(\xi)$';
    annotation('textbox', [0.05, 0.4, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

    % lg = legend('$\epsilon_{fwd; true}$', '$\epsilon_{fwd; model}$', '$\tau_{ODE}$', ['$\hat{\tau}_{ODE}(\lambda_{ODE}^h(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$'], ['$\hat{\tau}_{ODE}(\lambda_{ODE}^b(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$'], '$\epsilon_{d}^{max}$');
    lg = legend('$\epsilon_{fwd}^{true}$', '$\epsilon_{fwd}^{model}$', '$\mathcal{J}$', ['$\tilde{\mathcal{J}}(\lambda_{ODE}^h(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$'], ['$\tilde{\mathcal{J}}(\lambda_{ODE}^b(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$']);
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 3);
    lg.Position = [0.38, 0.02, 0.3, 0.07];


    path        = strcat(save_folder, '/qoi_fwd_error_cdf', '.fig');
    savefig(fg, path);
    saveas(fg, strrep(path, '.fig', '.png'));
end