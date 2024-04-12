clc; clear all; close all;
num_mcmc_samples = 100000;
lambda_val = linspace(1, 100, 1000); % For probabilistic bounds

%% Probability bound (Higham and Mary)
q_lam = compute_higham_prob("single", lambda_val);
lambda_zero = lambda_val(q_lam >= 0); lambda_zero = lambda_zero(1);
lambda_nine = lambda_val(q_lam >= 0.9); lambda_nine = lambda_nine(1);
lambda_one = lambda_val(q_lam == 1.0); lambda_one = lambda_one(1);
figure()
hold on
plot(lambda_val, q_lam, "Color", "k", "LineWidth", 3, "DisplayName", "$q(\lambda)$");
scatter(lambda_zero, 0, 180, "Marker", "square", "MarkerEdgeColor", "k", "MarkerFaceColor", "red", "DisplayName", "$\lambda_{0}: q(\lambda) = 0$")
scatter(lambda_nine, 0.9, 180, "Marker", "diamond", "MarkerEdgeColor", "k", "MarkerFaceColor", "green", "DisplayName", "$\lambda_{0.9}: q(\lambda) = 0.9$")
scatter(lambda_one, 1, 180, "Marker", "pentagram", "MarkerEdgeColor", "k", "MarkerFaceColor", "blue","DisplayName", "$\lambda_{1}: q(\lambda) = 1$")
hold off
grid on
box on
xlabel("$\lambda$", "Interpreter", "latex")
ylabel("$q(\lambda)$", "Interpreter", "latex")
title("Probability lower bound", "Interpreter", "latex", "FontSize", 20)
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
set(gca, 'XScale', 'log');
ylim([0, 1])
lg = legend();
set(lg, "Interpreter", "latex", "FontSize", 20)


for ii = 1:num_mcmc_samples
    fprintf('Iteration %d of %d (Percentage: %f)\n', ii, num_mcmc_samples, (ii/num_mcmc_samples)*100);
    samples = get_uniform_samples(2);
    true_sum = samples(1) + samples(2);
    sum_float = double(single(samples(1)) + single(samples(2)));
    % Relative Error
    abs_rel_error_true(ii) = compute_abs_rel_error(true_sum, sum_float);
    abs_rel_error_model_without_representation(ii) = compute_abs_rel_error_model(samples, "single", false);
    abs_rel_error_model_with_representation(ii) = compute_abs_rel_error_model(samples, "single", true);

    % Deterministic Forward Error Bound
    bound_det_without_representation(ii) = compute_deterministic_f_error_bound(samples, "single", false);
    bound_det_with_representation(ii) = compute_deterministic_f_error_bound(samples, "single", true);
    
    % Probabilistic Forward Error Bound
    bound_prob_without_representation(ii, :) = compute_probabilistic_f_error_bound(samples, "single", false, [lambda_zero, lambda_one]);
    bound_prob_with_representation(ii, :) = compute_probabilistic_f_error_bound(samples, "single", true, [lambda_zero, lambda_one]);

    % Backward Error
    bwd_error_true(ii) = compute_true_bwd_error(samples);
    bwd_error_model_without_representation(ii) = compute_modeled_bwd_error(samples, "single", false);
    bwd_error_model_with_representation(ii) = compute_modeled_bwd_error(samples, "single", true);
end

% Deterministic Backward Error Bound
bwd_error_deterministic_bound_without_representation = compute_deterministic_bwd_error_bound("single", false);
bwd_error_deterministic_bound_with_representation    = compute_deterministic_bwd_error_bound("single", true);

% Probabilistic Backward Error Bound
bwd_error_probabilistic_bound_without_representation = compute_probabilistic_bwd_error_bound("single", false, [lambda_zero, lambda_nine, lambda_one]);
bwd_error_probabilistic_bound_with_representation    = compute_probabilistic_bwd_error_bound("single", true, [lambda_zero, lambda_nine, lambda_one]);

figure()
subplot(1, 2, 1)
histogram(bwd_error_true, 100, 'Normalization', 'pdf', 'DisplayName', 'True')
hold on
h1 = histogram(bwd_error_model_without_representation, 100,  'Normalization', 'pdf', 'DisplayName', 'Model');
d1 = xline(bwd_error_deterministic_bound_without_representation, 'LineWidth', 5, 'LineStyle', '--', 'DisplayName', 'Deterministic Bound');
p1 = xline(bwd_error_probabilistic_bound_without_representation(1), 'LineWidth', 5, 'LineStyle', '-.', 'DisplayName', 'Probabilistic Bound ($\lambda_0$)' );
p1_2 = xline(bwd_error_probabilistic_bound_without_representation(2), 'LineWidth', 5, 'LineStyle', ':', 'DisplayName', 'Probabilistic Bound ($\lambda_{0.9}$)' );
d1.Color = "b"; p1.Color = "r"; p1_2.Color = "g";
hold off
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
box on;
xlabel('$e_{bwd}$', 'Interpreter', 'latex');
ylabel('$f_{e_{bwd}}(e_{bwd})$', 'Interpreter', 'latex');
title('Without Representation Error', 'Interpreter', 'latex', 'FontSize', 20)

subplot(1, 2, 2)
histogram(bwd_error_true, 100, 'Normalization', 'pdf', 'DisplayName', 'True')
hold on
h2 = histogram(bwd_error_model_with_representation, 100, 'Normalization', 'pdf', 'DisplayName', 'Model');
d2 = xline(bwd_error_deterministic_bound_with_representation, 'LineWidth', 5, 'LineStyle', '--', 'DisplayName', 'Deterministic Bound');
p2 = xline(bwd_error_probabilistic_bound_with_representation(1), 'LineWidth', 5, 'LineStyle', '-.', 'DisplayName', 'Probabilistic Bound ($\lambda_0$)' );
p2_2 = xline(bwd_error_probabilistic_bound_with_representation(2), 'LineWidth', 5, 'LineStyle', ':', 'DisplayName', 'Probabilistic Bound ($\lambda_{0.9}$)' );
d2.Color = "b"; p2.Color = "r"; p2_2.Color = "g";
hold off
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
box on;
xlabel('$e_{bwd}$', 'Interpreter', 'latex');
ylabel('$f_{e_{bwd}}(e_{bwd})$', 'Interpreter', 'latex');
title('With Representation Error', 'Interpreter', 'latex', 'FontSize', 20)



step = 1000;
num_steps = num_mcmc_samples / step;
for ii = 1:num_steps
    fprintf("(Mean) Iteration %d of %d (Percentage: %f)\n", ii, num_steps, (ii/num_steps)*100);
    mean_abs_rel_error_true(ii) = mean(abs_rel_error_true(1:ii*step));
    mean_abs_rel_error_model_with_representation(ii) = mean(abs_rel_error_model_with_representation(1:ii*step));
    mean_abs_rel_error_model_without_representation(ii) = mean(abs_rel_error_model_without_representation(1:ii*step));

    mean_bound_det_with_representation(ii) = mean(bound_det_with_representation(1:ii*step));
    mean_bound_det_without_representation(ii) = mean(bound_det_without_representation(1:ii*step));

    mean_bound_prob_with_representation(ii, :) = mean(bound_prob_with_representation(1:ii*step, :), 1);
    mean_bound_prob_without_representation(ii, :) = mean(bound_prob_without_representation(1:ii*step, :), 1);
end

f = figure();
set(f, "Position", [100, 100, 1500, 500])
subplot(1, 2, 1)
plot((1:num_steps)*step, mean_abs_rel_error_true, 'LineWidth', 3, 'DisplayName', 'True', 'Color', 'k');
hold on;
plot((1:num_steps)*step, mean_abs_rel_error_model_without_representation, 'LineWidth', 3, 'DisplayName', 'Modeled Relative Error', 'Color','b');
plot((1:num_steps)*step, mean_bound_det_without_representation, 'LineWidth', 3, 'DisplayName', 'Deterministic Bound', 'Color', 'b', 'LineStyle','-.');
plot((1:num_steps)*step, mean_bound_prob_without_representation(:, 1), 'LineWidth', 3, 'DisplayName', 'Probabilistic Bound $(\lambda_0$)', 'Color', 'b', 'LineStyle',':');
plot((1:num_steps)*step, mean_bound_prob_without_representation(:, 2), 'LineWidth', 3, 'DisplayName', 'Probabilistic Bound $(\lambda_1$)', 'Color', 'b', 'LineStyle','--');
hold off;
xlabel('Number of samples', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('Mean absolute relative error', 'Interpreter', 'latex', 'FontSize', 20);
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on
box on
ylim([1e-8, 1e-6])
title("Without Representation Error", "Interpreter", "latex");

subplot(1, 2, 2)
plot((1:num_steps)*step, mean_abs_rel_error_true, 'LineWidth', 3, 'DisplayName', 'True', 'Color', 'k');
hold on
plot((1:num_steps)*step, mean_abs_rel_error_model_with_representation, 'LineWidth', 3, 'DisplayName', 'Modeled Relative Error', 'Color','b');
plot((1:num_steps)*step, mean_bound_det_with_representation, 'LineWidth', 3, 'DisplayName', 'Deterministic Bound', 'Color', 'b', 'LineStyle','-.');
plot((1:num_steps)*step, mean_bound_prob_with_representation(:, 1), 'LineWidth', 3, 'DisplayName', 'Probabilistic Bound $(\lambda_0$)', 'Color', 'b', 'LineStyle',':');
plot((1:num_steps)*step, mean_bound_prob_with_representation(:, 2), 'LineWidth', 3, 'DisplayName', 'Probabilistic Bound $(\lambda_1$)', 'Color', 'b', 'LineStyle','--');
hold off;
xlabel('Number of samples', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('Mean absolute relative error', 'Interpreter', 'latex', 'FontSize', 20);
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on
box on
ylim([1e-8, 1e-6])
title("With Representation Error", "Interpreter", "latex");

figure()
histogram(abs_rel_error_true, 100, 'Normalization', 'pdf', 'DisplayName', 'True')


figure()
histogram(abs_rel_error_true, 100, 'Normalization', 'pdf', 'DisplayName', 'True');
hold on;
histogram(abs_rel_error_model_without_representation, 100, 'Normalization', 'pdf', 'DisplayName', 'Model (W/o Representation Error)');
histogram(abs_rel_error_model_with_representation, 100, 'Normalization', 'pdf', 'DisplayName', 'Model (W/ Representation Error)');
hold off
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
title('Emperical PDF for absolute forward error', 'Interpreter', 'latex', 'FontSize', 20);
xlabel("$\xi = \frac{|(x + y) - fl(x + y)|}{|x + y|}$", "Interpreter", "latex", "FontSize", 20)
ylabel("$f_{\Xi}(\xi)$", "Interpreter", "latex", "FontSize", 20)
box on


[F, X] = ecdf(abs_rel_error_true);
[F_model_without_rep, X_model_without_rep] = ecdf(abs_rel_error_model_without_representation);
[F_model_with_rep, X_model_with_rep] = ecdf(abs_rel_error_model_with_representation);

figure()
plot(X, F, 'LineWidth', 3, 'DisplayName', 'True');
hold on;
plot(X_model_without_rep, F_model_without_rep, 'LineWidth', 3, 'DisplayName', 'Model (W/o Representation Error)');
plot(X_model_with_rep, F_model_with_rep, 'LineWidth', 3, 'DisplayName', 'Model (W/ Representation Error)');
lg = legend();
set(lg, 'Interpreter', 'latex', 'FontSize', 20);
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 20);
grid on
title('Empirical CDF for absolute forward error', 'Interpreter', 'latex', 'FontSize', 20);
box on
xlabel("$\xi = \frac{|(x + y) - fl(x + y)|}{|x + y|}$", "Interpreter", "latex", "FontSize", 20)
ylabel("$F_{\Xi}(\xi)$", "Interpreter", "latex", "FontSize", 20)


%% Functions
function samples = get_uniform_samples(num_samples)
    samples = rand(num_samples, 1);
end

function samples = get_rel_error_samples(num_samples, precision)
    samples = rand(num_samples, 1);
    [meps, urd] = get_machine_info(precision);
    samples = urd*2.0*(samples - 0.5);
end

function [meps, urd] = get_machine_info(precision)
    if precision == "double"
        meps = eps("double");
        urd = eps("double") / 2;
    elseif precision == "single"
        meps = double(eps("single"));
        urd = meps / 2;
    elseif precision == "half"
        meps = double (eps("half"));
        urd = meps / 2;
    end
end

function abs_rel_error = compute_abs_rel_error(true_value, approx_value)
    true_val_p = convert_precision(true_value, "double");
    approx_value_p = convert_precision(approx_value, "double");
    abs_rel_error = abs(true_val_p - approx_value_p) / abs(true_val_p);
end

function conv_val = convert_precision(val, precision)
    if precision == "double"
        conv_val = double(val);
    elseif precision == "single"
        conv_val = single(val);
    elseif precision == "half"
        conv_val = half(val);
    else
        error("Invalid precision");
    end
end

function abs_rel_error = compute_abs_rel_error_model(samples, precision, representation_error)
    val_1 = samples(1);
    val_2 = samples(2);
    assert (isa(val_1, 'double'));
    assert (isa(val_2, 'double'));
    true_sum = val_1 + val_2;
   
    if representation_error
%         fprintf("Representation Error Included\n");
        rel_samples = get_rel_error_samples(3, precision);
        model_sum = (val_1*(1+rel_samples(1)) + val_2*(1+rel_samples(2)))*(1+rel_samples(3));
        abs_rel_error = abs(true_sum - model_sum) / abs(true_sum);
    else
%         fprintf("Representation Error NOT Included\n");
        rel_samples = get_rel_error_samples(1, precision);
        model_sum = (val_1 + val_2)*(1+rel_samples(1));
        abs_rel_error = abs(true_sum - model_sum) / abs(true_sum);
    end
    
end

function gamma_det = compute_deterministic_gamma(n, precision)
    [meps, urd] = get_machine_info(precision);
    gamma_det = (n*urd) / (1 - urd);
end

function gamma_prob = compute_probabilistic_gamma(n, precision, lambda_val)
    [meps, urd] = get_machine_info(precision);
    gamma_prob = lambda_val*sqrt(n)*urd;
end

function bound = compute_deterministic_f_error_bound(samples, precision, representation_error)
    val_1 = samples(1);
    val_2 = samples(2);
    assert (isa(val_1, 'double'));
    assert (isa(val_2, 'double'));
    true_sum = val_1 + val_2;
    if representation_error
        gamma_two = compute_deterministic_gamma(2, precision);
        bound = gamma_two*(abs(val_1) + abs(val_2)) / abs(true_sum);
    else
        gamma_one = compute_deterministic_gamma(1, precision);
        bound = gamma_one*(abs(val_1) + abs(val_2)) / abs(true_sum);
    end
end

function bound = compute_deterministic_bwd_error_bound(precision, representation_error)
    if representation_error
        bound = compute_deterministic_gamma(2, precision);
    else
        bound = compute_deterministic_gamma(1, precision);
    end
end

function bound = compute_probabilistic_bwd_error_bound(precision, representation_error, lambda_val)
    if representation_error
        bound = compute_probabilistic_gamma(2, precision, lambda_val);
    else 
        bound = compute_probabilistic_gamma(1, precision, lambda_val);
    end
end

function bound = compute_probabilistic_f_error_bound(samples, precision, representation_error, lambda_val)
    val_1 = samples(1);
    val_2 = samples(2);
    assert (isa(val_1, 'double'));
    assert (isa(val_2, 'double'));
    true_sum = val_1 + val_2;
    if representation_error
        gamma_two = compute_probabilistic_gamma(2, precision, lambda_val);
        bound = gamma_two.*(abs(val_1) + abs(val_2)) / abs(true_sum);
    else
        gamma_one = compute_probabilistic_gamma(1, precision, lambda_val);
        bound = gamma_one.*(abs(val_1) + abs(val_2)) / abs(true_sum);
    end
end

function q_lambda = compute_higham_prob(precision, lambda_val)
    % Since prob is independent for number of operations, same
    % probabilistic bounds are obtained with and woitout representation
    % error
 
    [meps, urd] = get_machine_info(precision);
    p_lambda = 1 - 2*exp( - ((lambda_val.*(1-urd)).^2) ./ 2.0 );
    q_lambda = 1 - 2*(1 - p_lambda);
end

function e_bwd = compute_true_bwd_error(samples)
    val_1 = samples(1);
    val_2 = samples(2);
    assert (isa(val_1, 'double'));
    assert (isa(val_2, 'double'));
    true_sum = val_1 + val_2;
    sum_float = double(single(samples(1)) + single(samples(2)));
    e_bwd = abs(true_sum - sum_float) / (abs(val_1) + abs(val_2));
end

function e_bwd = compute_modeled_bwd_error(samples, precision, representation_error)
    val_1 = samples(1);
    val_2 = samples(2);
    assert (isa(val_1, 'double'));
    assert (isa(val_2, 'double'));
    true_sum = val_1 + val_2;
    if representation_error        
        rel_samples = get_rel_error_samples(3, precision);
        model_sum = (val_1*(1+rel_samples(1)) + val_2*(1+rel_samples(2)))*(1+rel_samples(3));
    else
        rel_samples = get_rel_error_samples(1, precision);
        model_sum = (val_1 + val_2)*(1+rel_samples(1));
    end
    e_bwd = abs(true_sum - model_sum) / (abs(val_1) + abs(val_2));
end