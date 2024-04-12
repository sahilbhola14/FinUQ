void;
% Load Single Data
data_single = load('data_single.mat');
% Load Half Data
data_half = load('data_half.mat');


fg = figure();
set(fg, 'Position', [100 100 1000 800]);
[ha, pos] = tight_subplot(2,1,[.035 .03],[.2 .05],[.08 .01]);

axes(ha(1));
% Since hoeffding is independent of the sample size, we can plot it once
l1 = plot(data_half.test_confidence_levels, data_half.lambda_critical_hoeff(1, :), 'DisplayName', '$\lambda_{c}^h$');
l1.Color = 'k'; l1.Marker = 's'; l1.LineStyle = 'none'; l1.MarkerSize = 15; l1.MarkerFaceColor = l1.Color; l1.MarkerEdgeColor = 'k';l1.LineWidth = 2;
colors = get(gca,'colororder');
hold on;
for ii = 1:length(data_half.test_prob_size)
    l2 = plot(data_half.test_confidence_levels, data_half.lambda_critical_bern(ii, :), 'DisplayName', ['$\lambda_{c}^{b}(n = ', num2str(data_half.test_prob_size(ii)), ')$']);
    l2.Color = colors(ii, :); l2.Marker = '^'; l2.LineStyle = 'none'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';l2.LineWidth = 2;
end
hold off;
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'YScale', 'log', 'XScale', 'log', 'LineWidth', 1.5);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log', 'LineWidth', 1.5);
% lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'northwest', 'NumColumns', 2);
grid on; box on;
set(gca, 'XTickLabel', []);
% set(gca, 'XTick', linspace(0.9, 1, 5));
%xlabel('Confidence level, $\alpha$', 'Interpreter', 'latex', 'FontSize', 20);
ylim([1 4.0])
xlim([0 1.0])
set(gca, 'XMinorGrid', 'off');


axes(ha(2));
% Since hoeffding is independent of the sample size, we can plot it once
l1 = plot(data_single.test_confidence_levels, data_single.lambda_critical_hoeff(1, :), 'DisplayName', '$\lambda_{c}^h$');
l1.Color = 'k'; l1.Marker = 's'; l1.LineStyle = 'none'; l1.MarkerSize = 15; l1.MarkerFaceColor = l1.Color; l1.MarkerEdgeColor = 'k';l1.LineWidth = 2;
colors = get(gca,'colororder');
hold on;
for ii = 1:length(data_single.test_prob_size)
    l2 = plot(data_single.test_confidence_levels, data_single.lambda_critical_bern(ii, :), 'DisplayName', ['$\lambda_{c}^{b}(n = ', num2str(data_single.test_prob_size(ii)), ')$']);
    l2.Color = colors(ii, :); l2.Marker = '^'; l2.LineStyle = 'none'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';l2.LineWidth = 2;
end
hold off;
% set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'YScale', 'log', 'XScale', 'log', 'LineWidth', 1.5);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log', 'LineWidth', 1.5);
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'northwest', 'NumColumns', 2);
grid on; box on;
% xlabel('Confidence level, $\alpha$', 'Interpreter', 'latex', 'FontSize', 20);
% set(gca, 'XTick', linspace(0.9, 1, 5));
ylim([1 4.0])
xlim([0 1.0])
set(gca, 'XMinorGrid', 'off');

global_xlabel = 'Confidence level, $\alpha$';
annotation('textbox', [0.29, 0.099, 0.5, 0.05], 'String', global_xlabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

%global_ylabel = '$F_{\Xi}(\xi)$';
%annotation('textbox', [0.05, 0.4, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);
for ii = 1:length(data_half.test_prob_size)
    legend_entries{ii} = ['$\lambda_{c}^{b}(n = ', num2str(data_half.test_prob_size(ii)), ')$'];
end

lg = legend('$\lambda_c^h$', legend_entries{:});
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 4);
lg.Position = [0.38, 0.02, 0.3, 0.06];
saveas(gcf, 'lam_critical_compare.png');
