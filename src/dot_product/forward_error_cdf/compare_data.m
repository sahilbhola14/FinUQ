void;
test_confidence_levels = [0.9];
% Load Std data
data_std = load('data_StdNormal_single_vector_size_10000_num_samples_100000.mat');
% Load Uniform data
data_uniform = load('data_UMOnePOne_single_vector_size_10000_num_samples_100000.mat');


% Plot the data


fg = figure();
set(fg, 'Position', [100 100 800 700]);
[ha, pos] = tight_subplot(2,1,[.035 .03],[.25 .05],[.13 .03]);
% [ha, pos] = tight_subplot(2,1,[.035 .03],[.2 .05],[.08 .01]);
axes(ha(1));

[f_rel_error_true, x_rel_error_true] = ecdf(data_uniform.rel_error_true(1, :));
[f_rel_error_model, x_rel_error_model] = ecdf(data_uniform.rel_error_model(1, :));
[f_deterministic_fwd_bound, x_deterministic_fwd_bound] = ecdf(data_uniform.deterministic_fwd_bound(1, :));
[f_higham_fwd_bound, x_higham_fwd_bound] = ecdf(data_uniform.higham_fwd_bound(1, :, 1));
[f_berstein_fwd_bound, x_berstein_fwd_bound] = ecdf(data_uniform.bernstein_fwd_bound(1, :, 1));

line_set_width = 2.75;
l0 = plot(x_rel_error_true, f_rel_error_true, 'DisplayName', '$\epsilon_{fwd}^{true}$');
% l0.Color = 'r'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.Marker = '^'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
l0.Color = 'k'; l0.LineWidth = line_set_width; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';

hold on

l1 = plot(x_rel_error_model, f_rel_error_model, 'DisplayName', '$\epsilon_{fwd}^{model}$');
% l1.Color = 'k'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'v'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
l1.Color = 'r'; l1.LineWidth = line_set_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';


l2 = plot(x_deterministic_fwd_bound, f_deterministic_fwd_bound, 'DisplayName', '$\mathcal{C}_{D}\gamma_n$');
% l2.Color = 'b'; l2.LineWidth = 2; l2.MarkerFaceColor = l2.Color; l2.Marker = 'd'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
l2.Color = 'b'; l2.LineWidth = line_set_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 'none'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';

l3 = plot(x_higham_fwd_bound, f_higham_fwd_bound, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = 2; l3.MarkerFaceColor = l3.Color; l3.Marker = 'o'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = line_set_width; l3.MarkerFaceColor = l3.Color; l3.Marker = 'none'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';

l4 = plot(x_berstein_fwd_bound, f_berstein_fwd_bound, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 's'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';
l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = line_set_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'none'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

grid on; box on;
% lg = legend();
% set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southeast', 'NumColumns', 2);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.5)
xlim([1e-15, 1e+5])
set(gca, 'XTickLabel', []);



axes(ha(2));

[f_rel_error_true, x_rel_error_true] = ecdf(data_std.rel_error_true(1, :));
[f_rel_error_model, x_rel_error_model] = ecdf(data_std.rel_error_model(1, :));
[f_deterministic_fwd_bound, x_deterministic_fwd_bound] = ecdf(data_std.deterministic_fwd_bound(1, :));
[f_higham_fwd_bound, x_higham_fwd_bound] = ecdf(data_std.higham_fwd_bound(1, :, 1));
[f_berstein_fwd_bound, x_berstein_fwd_bound] = ecdf(data_std.bernstein_fwd_bound(1, :, 1));

line_set_width = 2.75;
l0 = plot(x_rel_error_true, f_rel_error_true, 'DisplayName', '$\epsilon_{fwd}^{true}$');
% l0.Color = 'r'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.Marker = '^'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
l0.Color = 'k'; l0.LineWidth = line_set_width; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';

hold on

l1 = plot(x_rel_error_model, f_rel_error_model, 'DisplayName', '$\epsilon_{fwd}^{model}$');
% l1.Color = 'k'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'v'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
l1.Color = 'r'; l1.LineWidth = line_set_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';


l2 = plot(x_deterministic_fwd_bound, f_deterministic_fwd_bound, 'DisplayName', '$\mathcal{C}_{D}\gamma_n$');
% l2.Color = 'b'; l2.LineWidth = 2; l2.MarkerFaceColor = l2.Color; l2.Marker = 'd'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
l2.Color = 'b'; l2.LineWidth = line_set_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 'none'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';

l3 = plot(x_higham_fwd_bound, f_higham_fwd_bound, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = 2; l3.MarkerFaceColor = l3.Color; l3.Marker = 'o'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = line_set_width; l3.MarkerFaceColor = l3.Color; l3.Marker = 'none'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';

l4 = plot(x_berstein_fwd_bound, f_berstein_fwd_bound, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 's'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';
l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = line_set_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'none'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

grid on; box on;
% lg = legend();
% set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southeast', 'NumColumns', 2);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.5)
xlim([1e-15, 1e+5])


global_xlabel = '$\xi = \frac{|\hat{y} - y|}{|y|}$';
annotation('textbox', [0.29, 0.15, 0.5, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

global_ylabel = '$F_{\Xi}(\xi)$';
annotation('textbox', [0.05, 0.45, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

lg = legend('$e_{fwd}^{true}$', '$e_{fwd}^{model}$', '$\mathcal{C}_{D}\gamma_n$', '$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^h(\alpha = 0.9))$', '$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^b(\alpha = 0.9))$');
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 4);
lg.Position = [0.4, 0.03, 0.3, 0.06];
saveas(gcf, 'cdf_compare.png');
