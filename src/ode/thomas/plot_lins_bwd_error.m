function plot_lins_bwd_error(lins_e_bwd_lp, lins_deterministic_e_bwd_bound, lins_higham_e_bwd_bound, lins_bernstein_e_bwd_bound, eval_num_intervals, lower_precision, prob_bound_confidence_levels, save_folder)
    mean_e_bwd = mean(lins_e_bwd_lp, 1);
    max_e_bwd = max(lins_e_bwd_lp, [], 1);
    fg = figure();
    set(fg, 'Position', [100 100 1000 500]);
    [ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .05],[.08 .015]);
    axes(ha(1));

    l0 = plot(eval_num_intervals, ones(1, length(eval_num_intervals)).*lins_deterministic_e_bwd_bound(1, 1), 'DisplayName', '$\gamma_{LS}$');
    l0.Color = 'b'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.MarkerSize = 15; l0.Marker = 'd'; l0.MarkerEdgeColor = 'k';
    hold on

    for ii = 1:length(prob_bound_confidence_levels)
        l1 = plot(eval_num_intervals, lins_higham_e_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_{LS}(\lambda_{LS}^h(\alpha = ', num2str(prob_bound_confidence_levels(ii)), '))$']);
        l1.Color = 'g'; l1.LineStyle = '--'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'o'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
        l2 = plot(eval_num_intervals, lins_bernstein_e_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_{LS}(\lambda_{LS}^b(\alpha = ', num2str(prob_bound_confidence_levels(ii)), '))$']);
        l2.Color = 'm'; l2.LineStyle = '-.'; l2.LineWidth = 2; l2.MarkerFaceColor = l2.Color; l2.Marker = 's'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
    end

    l3 = plot(eval_num_intervals, mean_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{mean}$');
    l3.Color = 'r'; l3.LineWidth = 2; l3.MarkerFaceColor = l3.Color; l3.Marker = '^'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
    l4 = plot(eval_num_intervals, max_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{max}$');
    l4.Color = 'k'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 'v'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

    hold off;
    ylim([1e-8, 1e-5])
    % Power of 2 ticks
    xticks([2.^(2:1:8)])
    xticklabels('$2^{' + string(2:1:8) + '}$')
    xlabel('$\#$ Intervals, $M$', 'Interpreter', 'latex')
    lg = legend();
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.5)
    grid on; box on;

    % Save the figure
    path        = strcat(save_folder, '/lins_bwd_error', '.fig');
    savefig(fg, path);
    saveas(fg, strrep(path, '.fig', '.png'));
end