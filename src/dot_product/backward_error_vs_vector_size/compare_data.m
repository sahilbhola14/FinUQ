void;
test_confidence_levels = [0.9];
% Load Std data
data_std = load('data_singlek_1_StdNormal_.mat');
% Load Uniform data
data_uniform = load('data_singlek_1_UMOnePOne_.mat');




line_set_width = 2.75;
fg = figure();
set(fg, 'Position', [100 100 800 700]);
% [ha, pos] = tight_subplot(2,1,[.01 .03],[.18 .1],[.08 .03]);
[ha, pos] = tight_subplot(2,1,[.035 .03],[.25 .05],[.1 .03]);
axes(ha(1));
colors = get(gca,'colororder');
l0 = plot(data_uniform.eval_vector_sizes, data_uniform.deterministic_bwd_bound, 'DisplayName', '$\gamma_n$'); 
l0.Color = 'b'; l0.LineWidth = line_set_width; l0.MarkerFaceColor = l0.Color; l0.MarkerSize = 15; l0.Marker = 'd'; l0.MarkerEdgeColor = 'k';
hold on
for ii = 1:length(data_uniform.test_confidence_levels)
    l1 = plot(data_uniform.eval_vector_sizes, data_uniform.higham_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(data_uniform.test_confidence_levels(ii)), '))$']);
    l1.Color = 'g'; l1.LineStyle = '--'; l1.LineWidth = line_set_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'o'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
    l2 = plot(data_uniform.eval_vector_sizes, data_uniform.bernstein_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(data_uniform.test_confidence_levels(ii)), '))$']);
    l2.Color = 'm'; l2.LineStyle = '-.'; l2.LineWidth = line_set_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 's'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
end
l3 = plot(data_uniform.eval_vector_sizes, data_uniform.mean_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{mean}$');
l3.Color = 'r'; l3.LineWidth = line_set_width; l3.MarkerFaceColor = l3.Color; l3.Marker = '^'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l4 = plot(data_uniform.eval_vector_sizes, data_uniform.max_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{max}$');
l4.Color = 'k'; l4.LineWidth = line_set_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'v'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

% lg = legend();
% set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.75)
grid on; box on;
hold off
set(gca, 'XTickLabel', []);
xlim([10, 10^6])
ylim([10^(-8), 1])
yticks([10^(-8), 10^(-6), 10^(-4), 10^(-2), 1])
set(gca, 'YMinorGrid', 'off');


axes(ha(2));
colors = get(gca,'colororder');
l0 = plot(data_std.eval_vector_sizes, data_std.deterministic_bwd_bound, 'DisplayName', '$\gamma_n$'); 
l0.Color = 'b'; l0.LineWidth = line_set_width; l0.MarkerFaceColor = l0.Color; l0.MarkerSize = 15; l0.Marker = 'd'; l0.MarkerEdgeColor = 'k';
hold on
for ii = 1:length(data_std.test_confidence_levels)
    l1 = plot(data_std.eval_vector_sizes, data_std.higham_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(data_std.test_confidence_levels(ii)), '))$']);
    l1.Color = 'g'; l1.LineStyle = '--'; l1.LineWidth = line_set_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'o'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
    l2 = plot(data_std.eval_vector_sizes, data_std.bernstein_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(data_std.test_confidence_levels(ii)), '))$']);
    l2.Color = 'm'; l2.LineStyle = '-.'; l2.LineWidth = line_set_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 's'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
end
l3 = plot(data_std.eval_vector_sizes, data_std.mean_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{mean}$');
l3.Color = 'r'; l3.LineWidth = line_set_width; l3.MarkerFaceColor = l3.Color; l3.Marker = '^'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l4 = plot(data_std.eval_vector_sizes, data_std.max_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{max}$');
l4.Color = 'k'; l4.LineWidth = line_set_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'v'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

lg = legend();
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.75)
grid on; box on;
hold off
xlim([10, 10^6])
ylim([10^(-8), 1])
yticks([10^(-8), 10^(-6), 10^(-4), 10^(-2), 1])
set(gca, 'YMinorGrid', 'off');

global_xlabel = 'Vector size, $n$';
annotation('textbox', [0.29, 0.15, 0.5, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

% global_ylabel = '$F_{\Xi}(\xi)$';
% annotation('textbox', [0.05, 0.45, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

lg = legend('$\gamma_n$', '$\tilde{\gamma}_n(\lambda_D^h(\alpha = 0.9))$', '$\tilde{\gamma}_n(\lambda_D^b(\alpha = 0.9))$', '$\epsilon_{bwd}^{mean}$', '$\epsilon_{bwd}^{max}$');
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 4);
lg.Position = [0.395, 0.03, 0.3, 0.06];
saveas(gcf, 'e_bwd_compare.png');