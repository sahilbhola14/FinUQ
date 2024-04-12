function plot_qoi_mean_abs_error_vs_samples(qoi_abs_error_true_mean, qoi_abs_error_model_mean, qoi_deterministic_e_fwd_bound_mean, ...
                                  qoi_higham_e_fwd_bound_mean, qoi_bernstein_e_fwd_bound_mean, eval_num_intervals, sample_mean_array, lower_precision, ...
                                  prob_bound_confidence_levels, save_folder)

    fg = figure();
    set(fg, 'Position', [100 1 1400 1200]);
    [ha, pos] = tight_subplot(3,2,[0.1 .07],[.18 .05],[.06 .015]);
    for ii =1:length(eval_num_intervals)
        axes(ha(ii));
        l1 = plot(sample_mean_array, qoi_abs_error_true_mean(:, ii), 'DisplayName', '$\epsilon_{fwd}^{mean}$');
        l1.Color = 'r'; l1.LineWidth = 2; l1.Marker = '^'; l1.MarkerSize = 15; l1.MarkerFaceColor = l1.Color; l1.MarkerEdgeColor = 'k';

        hold on;
        l2 = plot(sample_mean_array, qoi_abs_error_model_mean(:, ii), 'DisplayName', '$\epsilon_{fwd}^{model}$');
        l2.Color = 'k'; l2.LineWidth = 2; l2.Marker = 'v'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';

        l3 = plot(sample_mean_array, qoi_deterministic_e_fwd_bound_mean(:, ii), 'DisplayName', '$\tau_{ODE}$');
        l3.Color = 'b'; l3.LineWidth = 2; l3.Marker = 'd'; l3.MarkerSize = 15; l3.MarkerFaceColor = l3.Color; l3.MarkerEdgeColor = 'k';

        for jj = 1:length(prob_bound_confidence_levels)
            l4 = plot(sample_mean_array, qoi_higham_e_fwd_bound_mean(:, ii, jj), 'DisplayName', ['$\hat{\tau}_{ODE}(\lambda_{ODE}^h(\alpha = ', num2str(prob_bound_confidence_levels(jj)), '))$']);
            l4.Color = 'g'; l4.LineWidth = 2; l4.Marker = 'o'; l4.MarkerSize = 15; l4.MarkerFaceColor = l4.Color; l4.MarkerEdgeColor = 'k'; l4.LineStyle = '--';

            l5 = plot(sample_mean_array, qoi_bernstein_e_fwd_bound_mean(:, ii, jj), 'DisplayName', ['$\hat{\tau}_{ODE}(\lambda_{ODE}^b(\alpha = ', num2str(prob_bound_confidence_levels(jj)), '))$']);
            l5.Color = 'm'; l5.LineWidth = 2; l5.Marker = 's'; l5.MarkerSize = 15; l5.MarkerFaceColor = l5.Color; l5.MarkerEdgeColor = 'k'; l4.LineStyle = '-.';
        end
        hold off;
        title('$m$ = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)
        % xlabel('$\#$ Samples, $n$', 'Interpreter', 'latex', 'FontSize', 20)
        ylim([1e-8, 1e+0])
        yticks([1e-8,  1e-4, 1e+0])
        grid on; box on;

        set(gca, 'XScale', 'log', 'YScale', 'log');
        set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');

    end
    global_xlabel = '$\#$ Samples, n';
    annotation('textbox', [0.4, 0.085, 0.2, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    lg = legend('$\epsilon_{fwd; true}^{mean}$', '$\epsilon_{fwd;model}^{mean}$', '$\tau_{ODE}^{mean}$', ['$\hat{\tau}_{ODE}^{mean}(\lambda_{ODE}^h(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$'], ['$\hat{\tau}_{ODE}^{mean}(\lambda_{ODE}^b(\alpha = ', num2str(prob_bound_confidence_levels(end)), '))$']);
    set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 3);
    lg.Position = [0.35, 0.03, 0.3, 0.05];

    path        = strcat(save_folder, '/qoi_mean_fwd_error_vs_samples', '.fig');
    savefig(fg, path);
    saveas(fg, strrep(path, '.fig', '.png'));

end