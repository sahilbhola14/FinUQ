clc; clear all; close all;

% Begin User Input
run_config = "single";
rel_error_dist = "uniform";
num_mcmc_samples = 1000000;
% End User Input

% Get the config
model_config = get_config(run_config, rel_error_dist);
true_config = get_double_config(run_config, rel_error_dist);

% Global Constants
global initialize;
global reinitialize;
initialize = true;
reinitialize = true;

% Compute the sum
for ii = 1:num_mcmc_samples
    fprintf("Iteration %d of %d (Percentage: %f)\n", ii, num_mcmc_samples, (ii/num_mcmc_samples)*100);
    samples = get_uniform_samples(2);
    % Compute the true sum
    sum_true = compute_addition(samples, true_config);
    % Compute the model sum
    sum_model = compute_addition(samples, model_config);
    % Compute the sum in a given precision
    sum_precision = compute_addition_precision(samples, run_config);
    % Compute the relative error
    relative_error_model(ii) = compute_abs_relative_error(sum_true, sum_model);
    relative_error_precision(ii) = compute_abs_relative_error(sum_true, sum_precision);
end
if rel_error_dist == "uniform"
    if run_config == "single"
        save("relative_error_uniform_single.mat", "relative_error_model", "relative_error_precision");
    elseif run_config == "half"
        save("relative_error_uniform_half.mat", "relative_error_model", "relative_error_precision");
    else
        error("Invalid run_config");
    end
elseif rel_error_dist == "model"
    if run_config == "single"
        save("relative_error_model_single.mat", "relative_error_model", "relative_error_precision");
    elseif run_config == "half"
        save("relative_error_model_half.mat", "relative_error_model", "relative_error_precision");
    else
        error("Invalid run_config");
    end
else
    error("Invalid rel_error_dist");
end

% Plot the relative error histogram
figure()
hold on
histogram(relative_error_precision, 'Normalization', 'pdf', 'DisplayName', 'True')
histogram(relative_error_model, 'Normalization', 'pdf', 'DisplayName', 'Model')
lg = legend();
set(lg, 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
hold off
box on
xlabel("$\xi = \frac{|(x + y) - fl(x + y)|}{|(x+y)|}$", 'Interpreter', 'latex')
ylabel("$f(\xi)$", 'Interpreter', 'latex')
title("Emperical PDF of Relative Error", 'Interpreter', 'latex')
saveas(gcf, "relative_error_pdf.png")

% Plot the relative error cdf
[F, X] = ecdf(relative_error_precision);
[F_model, X_model] = ecdf(relative_error_model);
figure()
hold on
plot(X, F, 'LineWidth', 2, 'DisplayName', 'True')
plot(X_model, F_model, 'LineWidth', 2, 'DisplayName', 'Model')
lg = legend();
set(lg, 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
hold off
box on
grid on
xlabel("$\xi = \frac{|(x + y) - fl(x + y)|}{|(x+y)|}$", 'Interpreter', 'latex')
ylabel("$F(\xi)$", 'Interpreter', 'latex')
title("Emperical CDF of Relative Error", 'Interpreter', 'latex')
saveas(gcf, "relative_error_cdf.png")

% Error Convergence
step_size = 1000;
num_steps = num_mcmc_samples / step_size;
for ii = 1:num_steps
    mean_rel_error_precision(ii) = mean(relative_error_precision(1:ii*step_size));
    mean_rel_error_model(ii) = mean(relative_error_model(1:ii*step_size));
end

figure()
hold on
plot((1:num_steps)*step_size, mean_rel_error_precision, 'LineWidth', 2, 'DisplayName', 'True')
plot((1:num_steps)*step_size, mean_rel_error_model, 'LineWidth', 2, 'DisplayName', 'Model')
lg = legend();
set(lg, 'Interpreter', 'latex');
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
set(gca, 'YScale', 'log', 'XScale', 'log');
hold off
box on
grid on
xlabel("Number of Samples", 'Interpreter', 'latex')
ylabel("$\bar{\xi}$", 'Interpreter', 'latex')
title("Mean Relative Error", 'Interpreter', 'latex')
saveas(gcf, "mean_relative_error.png")

%% Functions
function config_double = get_double_config(run_config, rel_error_dist)
    keys_det = ["precision", "analysis"];
    vals_double = ["double", "off"];
    config_double = dictionary(keys_det, vals_double);
end

function config = get_config(run_config, rel_error_dist)
    keys_prob = ["precision", "analysis", "rel_error_dist"];
    keys_det = ["precision", "analysis"];

    vals_single_prob_uniform = ["single", "probabilistic", "uniform"];
    vals_half_prob_uniform = ["half", "probabilistic", "uniform"];
    vals_single_prob_model = ["single", "probabilistic", "model"];
    vals_half_prob_model = ["half", "probabilistic", "model"];
    vals_double = ["double", "off"];

    config_single_uniform = dictionary(keys_prob, vals_single_prob_uniform);
    config_single_model = dictionary(keys_prob, vals_single_prob_model);
    config_half_uniform = dictionary(keys_prob, vals_half_prob_uniform);
    config_half_model = dictionary(keys_prob, vals_half_prob_model);
    config_double = dictionary(keys_det, vals_double);
    if run_config == "single"
        if rel_error_dist == "uniform"
            config = config_single_uniform;
        elseif rel_error_dist == "model"
            config = config_single_model;
        else
            error("Invalid rel_error_dist");
        end
    elseif run_config == "half"
        if rel_error_dist == "uniform"
            config = config_half_uniform;
        elseif rel_error_dist == "model"
            config = config_half_model;
        else
            error("Invalid rel_error_dist");
        end

    elseif run_config == "double"
        config = config_double;
    else
        error("Invalid run_config");
    end
end

function val_p = convert_to_precision(val, precision)
    if precision == "single"
        val_p = single(val);
    elseif precision == "half"
        val_p = half(val);
    elseif precision == "double"
        val_p = double(val);
    else
        error("Invalid precision");
    end
end

function val = getGlobalInitialize
    global initialize
    val = initialize;
end

function val = getGlobalReinitialize
    global reinitialize
    val = reinitialize;
end

function [initialize, reinitialize] = getInitializeFlags
    initialize = getGlobalInitialize;
    reinitialize = getGlobalReinitialize;
end

function samples = get_uniform_samples(num_samples)
    samples = rand(num_samples, 1);
end

function addition = compute_addition(numbers, config)
    [initialize, reinitialize] = getInitializeFlags;
    a = DeltaOverload(numbers(1), config, initialize);
    b = DeltaOverload(numbers(2), config, initialize);
    addition_op = a + b;
    addition = addition_op.val;
    assert (isa(addition, "double"));
end

function addition = compute_addition_precision(numbers, precision)
    a = convert_to_precision(numbers(1), precision);
    b = convert_to_precision(numbers(2), precision);
    addition = a + b;
    addition = convert_to_precision(addition, "double");
    assert (isa(addition, "double"));
end

function abs_relative_error = compute_abs_relative_error(true_val, predicted_val)
    assert (isa(true_val, "double"));
    assert (isa(predicted_val, "double"));
    abs_relative_error = abs(true_val - predicted_val) ./ abs(true_val);
end
