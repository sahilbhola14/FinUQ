% Performs Scalar Multiplication when the numbers are sampled in the lower space 
void;

% Begin User Input
num_samples             = 100;
k                       = 1; % Vectors are sampled U[2^{k}, 2^{k+1}]
sample_space            = 'UMOnePOne'; % Valid Cases: 'UZeroOne', 'UMOnePOne', 'UPowerTwo', 'StdNormal'
eval_vector_sizes       = 10.^(1:1:6);
test_confidence_levels  = [0.5, 0.9, 0.99];
precision               = "single";
% End User Input

% Backward Error Bound
deterministic_bwd_bound = zeros(length(eval_vector_sizes), 1);
higham_bwd_bound        = zeros(length(eval_vector_sizes), length(test_confidence_levels));
bernstein_bwd_bound     = zeros(length(eval_vector_sizes), length(test_confidence_levels));
e_bwd                   = zeros(length(eval_vector_sizes), num_samples);

higham_critical_value    = zeros(length(eval_vector_sizes), length(test_confidence_levels));
bernstein_critical_value = zeros(length(eval_vector_sizes), length(test_confidence_levels));


for ii = 1:length(eval_vector_sizes)
    fprintf("Evaluating Vector Size: %d\n", eval_vector_sizes(ii));
    % Deterministic Bound
    deterministic_bwd_bound(ii)     = dot_product_bwd_bound_deterministic(eval_vector_sizes(ii), precision);
    % Higham Bound 
    [higham_bwd_bound(ii, :), higham_critical_value(ii, :), higham_prob(ii, :)]       = dot_product_bwd_bound_higham(eval_vector_sizes(ii), precision, test_confidence_levels);
    % Variance-informed bound
    [bernstein_bwd_bound(ii, :), bernstein_critical_value(ii, :), bern_prob(ii, :)] = dot_product_bwd_bound_bernstein(eval_vector_sizes(ii), precision, test_confidence_levels);
end
figure();
[ha, pos] = tight_subplot(1,1,[.01 .03],[.2 .09],[.15 .05]);
axes(ha(1));
colors = get(gca,'colororder');
for jj = 1:length(test_confidence_levels)
    l1 = plot(eval_vector_sizes, higham_critical_value(:, jj), 'DisplayName', sprintf('Higham Bound, %.2f', test_confidence_levels(jj)));
    l1.LineStyle = '-'; l1.LineWidth = 2.0; l1.Color = colors(jj, :);
    hold on;
    l2 = plot(eval_vector_sizes, bernstein_critical_value(:, jj), 'DisplayName', sprintf('Bernstein Bound, %.2f', test_confidence_levels(jj)));
    l2.LineStyle = '--'; l2.LineWidth = 2.0; l2.Color = colors(jj, :);
end
hold off
lg = legend();
xlabel('Vector Size, $n$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$\lambda_{D}$', 'Interpreter', 'latex', 'FontSize', 20);
set(lg, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log');
grid on; box on;


figure();
[ha, pos] = tight_subplot(1,1,[.01 .03],[.2 .09],[.15 .05]);
axes(ha(1));
test_lambdas = get_test_lambda();
colors = get(gca,'colororder');
for jj = 1:length(eval_vector_sizes)
    l1 = plot(test_lambdas, higham_prob(jj, :), 'DisplayName', sprintf('Higham Vector Size: %d', eval_vector_sizes(ii)));
    l1.LineStyle = '-'; l1.LineWidth = 2.0; l1.Color = colors(jj, :);
    hold on
    l2 = plot(test_lambdas, bern_prob(jj, :), 'DisplayName', sprintf('Bernstein Vector Size: %d', eval_vector_sizes(ii)));
    l2.LineStyle = '--'; l2.LineWidth = 2.0; l2.Color = colors(jj, :);
end
hold off
lg = legend();
set(lg, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log');
ylim([0, 1]);

figure();
[ha, pos] = tight_subplot(1,1,[.01 .03],[.2 .09],[.15 .05]);
axes(ha(1));
colors = get(gca,'colororder');
for ii = 1:length(test_confidence_levels)
    l1 = plot(eval_vector_sizes, higham_bwd_bound(:, ii), 'DisplayName', sprintf('Higham Bound, %.2f', test_confidence_levels(ii)));
    l1.LineStyle = '-'; l1.LineWidth = 2.0; l1.Color = colors(ii, :);
    hold on
    l2 = plot(eval_vector_sizes, bernstein_bwd_bound(:, ii), 'DisplayName', sprintf('Bernstein Bound, %.2f', test_confidence_levels(ii)));
    l2.LineStyle = '--'; l2.LineWidth = 2.0; l2.Color = colors(ii, :);
end
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log', 'YScale', 'log');
lg = legend();
set(lg, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 20);
grid on; box on;

figure();
error = higham_bwd_bound - bernstein_bwd_bound;
[ha, pos] = tight_subplot(1,1,[.01 .03],[.2 .09],[.15 .05]);
axes(ha(1));
colors = get(gca,'colororder');
for ii = 1:length(test_confidence_levels)
    l1 = plot(eval_vector_sizes, error(:, ii), "DisplayName", sprintf('Error, %.2f', test_confidence_levels(ii)));
    l1.LineStyle = '-'; l1.LineWidth = 2.0; l1.Color = colors(ii, :);
    hold on
end
l2 = plot(eval_vector_sizes, 1e-6*sqrt(eval_vector_sizes), 'DisplayName', 'Slope = 0.5');
l2.LineStyle = '--'; l2.Color = 'k'; l2.LineWidth = 2.0;
hold off
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20, 'XScale', 'log', 'YScale', 'log');
lg = legend();
set(lg, 'Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 20);
grid on; box on;

%% Functions
function samples = get_vector(n, precision, k)
    samples = rand(1, n)*(2^k) + (2^k); % U[2^k, 2^{k+1}]
    samples = convert_precision(samples, precision);
end

function samples = get_vector_Stdnorm(n, precision)
    samples = randn(1, n);  % Standard Normal
    samples = convert_precision(samples, precision);
end

function samples = get_vector_U_m_one_p_one(n, precision)
    samples = 2* (rand(1, n) - 0.5);  % U[-1, 1]
    samples = convert_precision(samples, precision);
end

function samples = get_vector_U_zero_one(n, precision)
    samples =  rand(1, n)          ;  % U[0, 1]
    samples = convert_precision(samples, precision);
end

function dot_product = compute_dot_product(vector_one, vector_two, precision)
    assert (length(vector_one) == length(vector_two));
    assert (isa(vector_one, precision) && isa(vector_two, precision));
    n = length(vector_one);
    summation = convert_precision(0, precision);
    for ii = 1:n
        summation = summation + vector_one(ii)*vector_two(ii);
    end
    dot_product = summation;
end

function e_bwd = compute_bwd_error(dot_product_dp, dot_product_lp, vector_one_dp, vector_two_dp)

    dot_product_dp  = convert_precision(dot_product_dp, 'double');
    dot_product_lp  = convert_precision(dot_product_lp, 'double');
    vector_one_dp   = convert_precision(vector_one_dp, 'double');
    vector_two_dp   = convert_precision(vector_two_dp, 'double');

    numerator       = abs(dot_product_dp - dot_product_lp);
    denominator     = abs(vector_one_dp)*abs(vector_two_dp)';
    e_bwd           = numerator / denominator;
end