void;

test_lambda             = logspace(0, 2, 100000);
test_prob_size          = [2, 3, 100, 1000, 10000, 100000];
test_confidence_levels  = [0.9, 0.95, 0.99];
precision       = "single";
gen_plots       = true;

%% Hoeffding vs Bernstein
lambda_critical_hoeff  = zeros(length(test_prob_size), length(test_confidence_levels));
lambda_critical_bern   = zeros(length(test_prob_size), length(test_confidence_levels));

for ii = 1:length(test_prob_size)
    prob_size = test_prob_size(ii);
    % Hoeffding based prob
    prob_hoeff                      = comp_hoeffding_prob(precision, test_lambda);
    lambda_critical_hoeff(ii, :)    = comp_lambda_critical(prob_hoeff, test_lambda, test_confidence_levels);

    % Bernstein based prob
    prob_bern                       = comp_bernstein_prob(prob_size, precision, test_lambda);
    lambda_critical_bern(ii, :)     = comp_lambda_critical(prob_bern, test_lambda, test_confidence_levels);

end

save(strcat('data_', convertStringsToChars(precision), '.mat'), "precision", "lambda_critical_bern", "lambda_critical_hoeff", "test_confidence_levels", "test_prob_size")
if gen_plots
    % Figure: Lambda critical vs confidence level
    fg = figure();
    set(fg, 'Position', [100 100 1000 500]);
    [ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .05],[.08 .01]);
    axes(ha(1));
    % Since hoeffding is independent of the sample size, we can plot it once
    l1 = plot(test_confidence_levels, lambda_critical_hoeff(1, :), 'DisplayName', '$\lambda_{c}^h$');
    l1.Color = 'k'; l1.Marker = 's'; l1.LineStyle = 'none'; l1.MarkerSize = 15; l1.MarkerFaceColor = l1.Color; l1.MarkerEdgeColor = 'k';l1.LineWidth = 2;

    colors = get(gca,'colororder');
    hold on;
    for ii = 1:length(test_prob_size)
        l2 = plot(test_confidence_levels, lambda_critical_bern(ii, :), 'DisplayName', ['$\lambda_{c}^{b}(n = ', num2str(test_prob_size(ii)), ')$']);
        l2.Color = colors(ii, :); l2.Marker = '^'; l2.LineStyle = 'none'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';l2.LineWidth = 2;
    end
    hold off;
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'YScale', 'log', 'XScale', 'log', 'LineWidth', 1.5);
    lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'northwest', 'NumColumns', 2);
    grid on; box on;
    xlabel('Confidence level, $\alpha$', 'Interpreter', 'latex', 'FontSize', 20);
    ylim([-inf 5.0])
    xlim([0 1.0])
    saveas(gcf, 'lam_critical.png');
end

%% Bersntein vs Deterministic bound

test_confidence_levels  = [0.9, 0.95, 0.99, 1.0];
test_lambda             = logspace(0, 2, 1000);
lambda_critical_hoeff  = zeros(length(test_prob_size), length(test_confidence_levels));
lambda_critical_bern   = zeros(length(test_prob_size), length(test_confidence_levels));
urd = comp_urd(precision);
problem_size_max = 1000;
for n = 1:problem_size_max
    % Hoeffding 
    prob_hoeff                      = comp_hoeffding_prob(precision, test_lambda);
    lambda_critical_hoeff(n, :)    = comp_lambda_critical(prob_hoeff, test_lambda, test_confidence_levels);
    hoeff_bound(n, :) = lambda_critical_hoeff(n, :)*sqrt(n)*urd;

    % Bernstein
    prob_bern                       = comp_bernstein_prob(n, precision, test_lambda);
    lambda_critical_bern(n, :)      = comp_lambda_critical(prob_bern, test_lambda, test_confidence_levels);
    bern_bound(n, :) = lambda_critical_bern(n, :)*sqrt(n)*urd;

    % Deterministic
    deterministic_bound(n) = (n*urd)/(1-(n*urd));
end

for ii = 1:length(test_confidence_levels)
    problem_array = 1:problem_size_max;
    condition = bern_bound(:, ii) < deterministic_bound(:);
    critical_n_bern(ii) = min(problem_array(condition));
    condition = hoeff_bound(:, ii) < deterministic_bound(:);
    critical_n_hoeff(ii) = min(problem_array(condition));

end
fprintf('Critical n for bernstein: %d\n', critical_n_bern);
fprintf('Critical n for hoeffding: %d\n', critical_n_hoeff);

% save('bern_vs_deter_critical_n.mat', 'bern_bound', 'deterministic_bound', 'critical_n', 'test_confidence_levels', 'problem_size_max', 'test_lambda', 'lambda_critical_bern', 'urd')

fg = figure();
set(fg, 'Position', [100 100 1000 500]);
[ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .05],[.08 .01]);
axes(ha(1));
l1 = plot(test_confidence_levels, critical_n_hoeff, 'DisplayName', '$n_{h}^c$');
l1.Color = 'k'; l1.Marker = 's'; l1.LineStyle = 'none'; l1.MarkerSize = 15; l1.MarkerFaceColor = l1.Color; l1.MarkerEdgeColor = 'k';l1.LineWidth = 2;
hold on
l2 = plot(test_confidence_levels, critical_n_bern, 'DisplayName', '$n_{b}^c$');
l2.Color = 'r'; l2.Marker = '^'; l2.LineStyle = 'none'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';l2.LineWidth = 2;
hold off
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'YScale', 'log', 'XScale', 'log');
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'northwest');
grid on; box on;
xlabel('Confidence level, $\alpha$', 'Interpreter', 'latex', 'FontSize', 20);
ylim([-inf 100])
xlim([0 1.0])
saveas(gcf, 'n_critical.png');


%figure()
%plot(1:problem_size_max, deterministic_bound, 'DisplayName', '$\gamma_{n}$', 'LineWidth', 2, 'Color', 'k');
%hold on
%colors = get(gca,'colororder');
%for ii = 1:length(test_confidence_levels)
%    l1 =  plot(1:problem_size_max, bern_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n;\lambda^b_c(\alpha = ', num2str(test_confidence_levels(ii)), ')$'], 'LineWidth', 2);
%    l1.Color = colors(ii, :);
%    l2 = plot(1:problem_size_max, hoeff_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n;\lambda^h_c(\alpha = ', num2str(test_confidence_levels(ii)), ')$'], 'LineWidth', 2);
%    l2.Color = colors(ii, :); l2.LineStyle = '--';
%end
%xline(critical_n_bern(end), 'DisplayName', '$n_{b}^c$', 'LineWidth', 4, 'LineStyle', '--', 'Color', 'k');
%xline(critical_n_hoeff(end), 'DisplayName', '$n_{h}^c$', 'LineWidth', 4, 'LineStyle', '-.', 'Color', 'k');
%hold off
%set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'YScale', 'log', 'XScale', 'log');
%grid on; box on;
%lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'southeast');
%xlabel('Problem size, $n$', 'Interpreter', 'latex', 'FontSize', 20);