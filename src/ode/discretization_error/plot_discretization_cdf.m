function plot_discretization_cdf(e_d_true, lower_bound_e_d, upper_bound_e_d, eval_num_intervals)
    fg = figure();
    set(fg, 'Position', [100 1 1400 1200]);
    % [ha, pos] = tight_subplot(3,2,[0.13 .1],[.18 .05],[.075 .01]);
    [ha, pos] = tight_subplot(3,2,[0.1 .07],[.18 .05],[.1 .015]);
    set_line_width = 2.75;
    for ii =1:length(eval_num_intervals)
        axes(ha(ii));
        [f_true, x_true] = ecdf(e_d_true{ii});
        [lower_f, lower_x] = ecdf(lower_bound_e_d{ii});
        [upper_f, upper_x] = ecdf(upper_bound_e_d{ii});
        l0 = plot(x_true, f_true, 'DisplayName', '$\epsilon_{d}^{true}$');
        l0.Color = 'k'; l0.LineWidth = set_line_width; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
        hold on;
        % l1 = plot(lower_x, lower_f, 'DisplayName', '$\epsilon_{d}^{lower}$');
        % l1.Color = 'r'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'r';

        l2 = plot(upper_x, upper_f, 'DisplayName', '$\epsilon_{d}^{upper}$');
        l2.Color = 'k'; l2.LineWidth = set_line_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 'none'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'b'; l2.LineStyle = '--';

        hold off;

        ylim([1e-4, 1]);
        xlim([1e-8, 1e+0])

        title('$M$ = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)

        box on;
        grid on;
        % grid on; box on;

        set(gca, 'XScale', 'log');
        set(gca, 'YScale', 'log');
        set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
        set(gca, 'XMinorGrid', 'off', 'YMinorGrid', 'off');
        set(gca, 'LineWidth', 1.5);

    end

    % global_xlabel = '$\xi = u(x_i) - \tilde{u}(x_i)$';
    global_xlabel = '$\xi = u(x_i) - \tilde{u}(x_i)$';
    annotation('textbox', [0.43, 0.085, 0.2, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    global_ylabel = '$F_{\Xi}(\xi)$';
    annotation('textbox', [0.05, 0.4, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

    lg = legend('$\epsilon_{d}$', '$\mathbf{A}^{-1}\mathbf{t}^{inf}$');
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 3);
    lg.Position = [0.38, 0.03, 0.3, 0.05];

    path        = strcat('./discretization_cdf', '.fig');
    savefig(fg, path);
    saveas(fg, strrep(path, '.fig', '.png'));
end