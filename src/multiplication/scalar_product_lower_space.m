% Performs Scalar Multiplication when the numbers are sampled in the lower space 
% Numbers are sampled from U[2^{k}, 2^{k+1}] 
void;
% Begin User Input
num_mcmc_samples    = 1000000;
sample_space        = "single";
k                   = 6;
% End User Input
urd = compute_unit_round_off(sample_space);
%% Summation
for ii = 1:num_mcmc_samples
    fprintf('Percent Complete: %f\n', ii/num_mcmc_samples*100)
    numbers_samples     = get_samples_given_precision(2, sample_space, k);
    numbers_samples_dp  = convert_precision(numbers_samples, 'double');
    % True Summation
    summation_dp        = compute_product(numbers_samples_dp);
    % Lower Precisin Summation
    summation_lp        = compute_product(numbers_samples);
    % Model Summation
    rel_error_model_sample = get_relative_error_samples(1, sample_space);
    summation_model     = summation_dp*(1 + rel_error_model_sample);
    % Error
    abs_error_true(ii)       = compute_absolute_error(summation_dp, summation_lp);
    rel_error_true(ii)       = compute_relative_error(summation_dp, summation_lp);
    rel_error_model(ii)      = compute_relative_error(summation_dp, summation_model);
end

%% Figure: Error PDF
figure()
subplot(1, 2, 1)
histogram(abs_error_true, 'Normalization', 'pdf', 'DisplayName', 'Model', 'NumBins', 50)
hold on
xline(urd, 'LineWidth', 5, 'Color', 'k', 'LineStyle', '--', 'DisplayName', '$u$')
hold off
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
xlabel('$\xi = \vert (\hat{a} \otimes \hat{b}) - (\hat{a} \times \hat{b}) \vert$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$f_{\Xi}(\xi)$', 'Interpreter', 'latex', 'FontSize', 20)
title('Absolute Error PDF', 'Interpreter', 'latex', 'FontSize', 20)
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20)
box on

subplot(1, 2, 2)
histogram(rel_error_true, 'Normalization', 'pdf', 'DisplayName', 'True', 'NumBins', 50)
hold on
histogram(rel_error_model, 'Normalization', 'pdf', 'DisplayName', 'Model', 'NumBins', 50)
xline(urd, 'LineWidth', 5, 'Color', 'k', 'LineStyle', '--', 'DisplayName', '$u$')
xline(0.5*urd, 'LineWidth', 5, 'Color', '#A2142F', 'LineStyle', '--', 'DisplayName', '$u/2$')
hold off
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
xlabel('$\xi = \vert \frac{(\hat{a} \otimes \hat{b}) - (\hat{a} \times \hat{b})}{(\hat{a} \times \hat{b})} \vert$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$f_{\Xi}(\xi)$', 'Interpreter', 'latex', 'FontSize', 20)
title('Relative Error PDF', 'Interpreter', 'latex', 'FontSize', 20)
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20)
box on
sgtitle(['$\hat{a}, \hat{b} \sim \mathcal{S}(\mathcal{U}[2^{', num2str(k), '}, 2^{', num2str(k+1), '}])$'], 'Interpreter', 'latex', 'FontSize', 20)
savefig(['Figures_without_representation_error/pdf_scalar_addition_lower_space_k_', num2str(k), '_num_samples_', num2str(num_mcmc_samples), '_space_', convertStringsToChars(sample_space), '.fig'])


%% Figure: Error CDF
[F, X] = ecdf(rel_error_true);
[F_model, X_model] = ecdf(rel_error_model);
save(['cdf_scalar_product_k_', num2str(k), '_num_samples_', num2str(num_mcmc_samples), '_space_', convertStringsToChars(sample_space), '.mat'], 'F', 'X', 'F_model', 'X_model')
figure()
plot(X, F, 'LineWidth', 4, 'DisplayName', 'True')
hold on
plot(X_model, F_model, 'LineWidth', 4, 'DisplayName', 'Model')
hold off
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
xlabel('$\xi = \vert \frac{(\hat{a} \otimes \hat{b}) - (\hat{a} + \hat{b})}{(\hat{a} + \hat{b})} \vert$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$F_{\Xi}(\xi)$', 'Interpreter', 'latex', 'FontSize', 20)
title(['Relative Error CDF', '$(\hat{a}, \hat{b} \sim \mathcal{S}(\mathcal{U}[2^{', num2str(k), '}, 2^{', num2str(k+1), '}]))$'], 'Interpreter', 'latex', 'FontSize', 20)
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20)
grid on
box on
savefig(['Figures_without_representation_error/cdf_scalar_addition_lower_space_k_', num2str(k), '_num_samples_', num2str(num_mcmc_samples), '_space_', convertStringsToChars(sample_space), '.fig'])

%% Figure: Error vs. Samples
step = 100;
assert (num_mcmc_samples > step);
assert (mod(num_mcmc_samples, step) == 0);
num_steps = floor(num_mcmc_samples/step);
for ii = 1:num_steps
    mean_rel_error_true(ii)  = mean(rel_error_true(1:ii*step));
    mean_rel_error_model(ii) = mean(rel_error_model(1:ii*step));
end
figure()
plot((1:num_steps)*step, mean_rel_error_true, 'LineWidth', 4, 'DisplayName', 'True')
hold on
plot((1:num_steps)*step, mean_rel_error_model, 'LineWidth', 4, 'DisplayName', 'Model')
hold off
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
lg = legend(); set(lg, 'Interpreter', 'latex', 'FontSize', 20)
grid on
box on
set(gca, 'XScale', 'log', 'YScale', 'log')
xlabel('Number of Samples', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$\overline{\xi}$', 'Interpreter', 'latex', 'FontSize', 20)
title(['Mean Relative Error vs. Number of Samples', '$(\hat{a}, \hat{b} \sim \mathcal{S}(\mathcal{U}[2^{', num2str(k), '}, 2^{', num2str(k+1), '}]))$'], 'Interpreter', 'latex', 'FontSize', 20)
savefig(['Figures_without_representation_error/mean_error_scalar_addition_lower_space_k_', num2str(k), '_num_samples_', num2str(num_mcmc_samples), '_space_', convertStringsToChars(sample_space), '.fig'])


%% Functions
function samples = get_samples_given_precision(n, precision, k)
    samples = rand(1, n)*(2^k) + (2^k); % U[2^k, 2^{k+1}]
    samples = convert_precision(samples, precision);
end

function samples = get_relative_error_samples(n, precision)
    urd = compute_unit_round_off(precision);
    samples = urd*2.0*(rand(1, n) - 0.5);
end

function summation = compute_product(numbers)
    a = numbers(1); b = numbers(2);
    summation = a * b;
end

function val_conv = convert_precision(val, precision)

    if precision == 'double'
        val_conv = double(val);
    elseif precision == 'single'
        val_conv = single(val);
    elseif precision == 'half'
        val_conv = half(val);
    else
        error('Invalid precision');
    end

end

function rel_error = compute_relative_error(true_val, model_val)

    true_val   = convert_precision(true_val, 'double');
    model_val  = convert_precision(model_val, 'double');

    assert (isa(true_val, 'double') && isa(model_val, 'double'));

    rel_error = abs(true_val - model_val) / abs(true_val);

end

function abs_error = compute_absolute_error(true_val, model_val)

    true_val   = convert_precision(true_val, 'double');
    model_val  = convert_precision(model_val, 'double');

    assert (isa(true_val, 'double') && isa(model_val, 'double'));

    abs_error = abs(true_val - model_val);

end

function urd = compute_unit_round_off(precision)
        if precision == "double"
            urd = eps('double');
            urd = urd / 2;
        elseif precision == "single"
            urd = double(eps('single'));
            urd = urd / 2;
        elseif precision == "half"
            urd = double(eps('half'));
            urd = urd / 2;
        else
            error('Invalid precision');
        end
end