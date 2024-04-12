void;

%% Begin User Inputs
eval_parameter_samples          = 10000;                            % Number of parameter samples
eval_num_intervals              = 2.^(2:7);                         % Number of intervals for the discretization of the spatial domain (ODE)
verbose                         = false;                            % Verbose output
checks                          = true;                             % Check the computation for inf and NaN
lower_precision                 = "single";                           % Lower precision for the computation (Used to enforce no representation error)
higher_precision                = "double";                         % Higher precision for the computation (For comparison) 
prob_bound_confidence_levels    = [0.99];                          % Confidence levels for the probabilistic bounds
converged_mean_anal_p           = load("anal_p_convergence.mat");   % Obtained from the script "comp_analytical_integral_convergence.m"
re_plot                         = true;                            % Re-plot the results from the saved data
step_for_mean_convergence       = 100;                              % Step for the mean convergence (Mean of solution is computed every 'step_for_mean_convergence' steps)
num_experiments                 = 1;
plot_data_file                  = 'data_exp_1e5_samples';
%% End User Inputs

global save_folder;
save_folder = lower_precision;
if not(isfolder(save_folder))
    mkdir(save_folder)
end

% Converged solution of the integral quantity
qoi_true_sol = converged_mean_anal_p.mean_anal_p(end);


%%%%%%%%%%%%%%%%%
%% Computation %%
%%%%%%%%%%%%%%%%%

%% Backward bounds for the Tri-diagonal system solve
lins_deterministic_e_bwd_bound    = comp_lins_bwd_bound_deterministic(lower_precision); % Deterministic backward bounds for the tri-diagonal sys.
lins_higham_e_bwd_bound           = comp_lins_bwd_bound_higham(eval_num_intervals, lower_precision, prob_bound_confidence_levels); % Higham's backward bounds for the tri-diagonal sys.
lins_bernstein_e_bwd_bound        = comp_lins_bwd_bound_bernstein(eval_num_intervals, lower_precision, prob_bound_confidence_levels); % Bernstein's backward bounds for the tri-diagonal sys.

for iexp = 1:num_experiments
    fprintf("Experiment %d/%d\n", iexp, num_experiments);
    %% Parameters
    if re_plot
        assert(num_experiments == 1, 'Re-plotting is only supported for a single experiment');
        load(get_save_path("eval_parameters"), "parameters");
    else
        parameters = sample_parameters(eval_parameter_samples);
        save(get_save_path("eval_parameters"), "parameters");
    end

    %% Comp: Compute Solutions
    solution_size                       = [eval_parameter_samples, length(eval_num_intervals)];

    qoi_analytical                      = zeros(solution_size); % Analytical solution for the integral QoI
    qoi_hp                              = zeros(solution_size); % Higher precision solution for the integral QoI
    qoi_lp                              = zeros(solution_size); % Lower precision solution for the integral QoI
    qoi_model                           = zeros(solution_size); % Model solution for the integral QoI (Model for relative error used in the standard arithmetic model)

    lins_e_bwd_hp                       = zeros(solution_size); % Backward error for the higher precision solution
    lins_e_bwd_lp                       = zeros(solution_size); % Backward error for the lower precision solution
    lins_e_bwd_model                    = zeros(solution_size); % Backward error for the model solution (Model for relative error used in the standard arithmetic model)

    lins_e_fwd_hp                       = zeros(solution_size); % Forward error for the higher precision solution of the linear system solve
    lins_e_fwd_lp                       = zeros(solution_size); % Forward error for the lower precision solution of the linear system solve

    condition_hp                        = zeros(solution_size); % Condition number for the higher precision solution of the linear system solve (forward error \approx condition number * backward error)
    condition_lp                        = zeros(solution_size); % Condition number for the lower precision solution of the linear system solve (forward error \approx condition number * backward error)

    discretization_max_error_bound      = zeros(solution_size); % Maximum discretization error for the analytical solution
    true_discretization_error           = zeros(solution_size); % True Numerical Discretization Error 

    qoi_abs_error_true                  = zeros(solution_size); % Absolute error of the QoI between the higher and lower precision solutions
    qoi_abs_error_model                 = zeros(solution_size); % Absolute error of the QoI between the higher precision and model solutions

    qoi_rel_error_true                  = zeros(solution_size); % Relative error of the QoI between the higher and lower precision solutions
    qoi_rel_error_model                 = zeros(solution_size); % Relative error of the QoI between the higher precision and model solutions

    lins_abs_error_true                 = zeros(solution_size); % Absolute error of the linear system solution between the higher and lower precision solutions
    lins_abs_error_model                = zeros(solution_size); % Absolute error of the linear system solution between the higher precision and model solutions

    lins_abs_sum_lp                     = zeros(solution_size); % Absolute sum of the linear system solution for the lower precision solution

    if re_plot == false
        for ii = 1:eval_parameter_samples
            parameter_sample = parameters(ii, :);
            if mod(ii, 100) == 0
                fprintf("Computing solution %d/%d\n", ii, eval_parameter_samples);
            end

            parfor jj = 1:length(eval_num_intervals)
                num_intervals           = eval_num_intervals(jj);
                % Analytical Solution (Qoi)
                qoi_analytical(ii, jj)  = comp_analytical_integral(parameter_sample);

                % Analytical Solution (Linear System)
                lins_analytical = comp_analytical_state(parameter_sample, num_intervals);

                % Higher Precision Solution (w/o representation error)
                [p, lins_hp, e_bwd, e_fwd, C] = comp_ode_sol_wo_rep_error(num_intervals, parameter_sample, higher_precision, lower_precision, verbose, checks);
                qoi_hp(ii, jj) = p; lins_e_bwd_hp(ii, jj) = e_bwd; lins_e_fwd_hp(ii, jj) = e_fwd; condition_hp(ii, jj) = C;

                % Lower Precision Solution (w/o representation error)
                [p, lins_lp, e_bwd, e_fwd, C] = comp_ode_sol_wo_rep_error(num_intervals, parameter_sample, lower_precision, lower_precision, verbose, checks);
                qoi_lp(ii, jj) = p; lins_e_bwd_lp(ii, jj) = e_bwd; lins_e_fwd_lp(ii, jj) = e_fwd; condition_lp(ii, jj) = C;

                % Model Solution
                [p, lins_model, e_bwd] = comp_ode_sol_model_rel_error(num_intervals, parameter_sample, higher_precision, lower_precision, verbose, checks);
                qoi_model(ii, jj) = p; lins_e_bwd_model(ii, jj) = e_bwd;

                % Analytical Numerical Error Due to Discretization
                [e_disc_sup, e_disc_inf, e_disc_max]    = comp_analytical_discretization_error_bounds(parameter_sample, num_intervals);
                discretization_max_error_bound(ii, jj)        = e_disc_max;
                true_discretization_error(ii, jj)   = comp_true_numerical_discretization_error(lins_analytical, lins_hp);
                % true_discretization_error(ii, jj)

                % Absolute Error QoI
                qoi_abs_error_true(ii, jj)      = comp_abs_error(qoi_hp(ii, jj), qoi_lp(ii, jj));
                qoi_abs_error_model(ii, jj)     = comp_abs_error(qoi_hp(ii, jj), qoi_model(ii, jj));

                % Relative Error QoI
                qoi_rel_error_true(ii, jj)      = comp_rel_error(qoi_hp(ii, jj), qoi_lp(ii, jj));
                qoi_rel_error_model(ii, jj)     = comp_rel_error(qoi_hp(ii, jj), qoi_model(ii, jj));

                % Absolute Error Lins solution
                lins_abs_error_true(ii, jj)     = comp_lins_abs_error(lins_hp, lins_lp); 
                lins_abs_error_model(ii, jj)    = comp_lins_abs_error(lins_hp, lins_model);

                % Absolute sum of the Linear system solution
                lins_abs_sum_lp(ii, jj)         = comp_lins_abs_sum(lins_lp); 

            end
        end

        % Save the results
        save(get_save_path(strcat("data_exp_", num2str(iexp))), "eval_parameter_samples", "eval_num_intervals", "prob_bound_confidence_levels", "qoi_true_sol", "higher_precision", ...
            "lower_precision", "qoi_analytical", "qoi_hp", "qoi_lp", "qoi_model", "lins_e_bwd_hp", "lins_e_bwd_lp", "lins_e_bwd_model", "lins_e_fwd_hp", ...
            "lins_e_fwd_lp", "condition_hp", "condition_lp", "discretization_max_error_bound", "true_discretization_error", "qoi_abs_error_true", "qoi_abs_error_model", "qoi_rel_error_true", ...
            "qoi_rel_error_model", "lins_abs_error_true", "lins_abs_error_model", "lins_abs_sum_lp")
    else
        fprintf("Loading the data from '%s'\n", get_save_path(plot_data_file));
        % Load the Data
        data = load(get_save_path(plot_data_file));

        eval_parameter_samples          = get_data_from_struc(data, "eval_parameter_samples");
        eval_num_intervals              = get_data_from_struc(data, "eval_num_intervals");
        prob_bound_confidence_levels    = get_data_from_struc(data, "prob_bound_confidence_levels");
        qoi_true_sol                    = get_data_from_struc(data, "qoi_true_sol");
        higher_precision                = get_data_from_struc(data, "higher_precision");
        lower_precision                 = get_data_from_struc(data, "lower_precision");

        qoi_analytical                  = get_data_from_struc(data, "qoi_analytical");
        qoi_hp                          = get_data_from_struc(data, "qoi_hp");
        qoi_lp                          = get_data_from_struc(data, "qoi_lp");
        qoi_model                       = get_data_from_struc(data, "qoi_model");

        lins_e_bwd_hp                   = get_data_from_struc(data, "lins_e_bwd_hp"); 
        lins_e_bwd_lp                   = get_data_from_struc(data, "lins_e_bwd_lp"); 
        lins_e_bwd_model                = get_data_from_struc(data, "lins_e_bwd_model"); 

        lins_e_fwd_hp                   = get_data_from_struc(data, "lins_e_fwd_hp");
        lins_e_fwd_lp                   = get_data_from_struc(data, "lins_e_fwd_lp"); 

        condition_hp                    = get_data_from_struc(data, "condition_hp"); 
        condition_lp                    = get_data_from_struc(data, "condition_lp"); 

        discretization_max_error_bound        = get_data_from_struc(data, "discretization_max_error");  % This is a mistake in the variable name in the old data (Comment out in when not available)
        % discretization_max_error_bound  = get_data_from_struc(data, "discretization_max_error_bound");  % Uncomment when available
        % true_discretization_error       = get_data_from_struc(data, "true_discretization_error"); % This is not available in the old data (Uncomment when available)

        qoi_abs_error_true              = get_data_from_struc(data, "qoi_abs_error_true"); 
        qoi_abs_error_model             = get_data_from_struc(data, "qoi_abs_error_model"); 

        qoi_rel_error_true              = get_data_from_struc(data, "qoi_rel_error_true"); 
        qoi_rel_error_model             = get_data_from_struc(data, "qoi_rel_error_model"); 

        lins_abs_error_true             = get_data_from_struc(data, "lins_abs_error_true"); 
        lins_abs_error_model            = get_data_from_struc(data, "lins_abs_error_model"); 

        lins_abs_sum_lp                 = get_data_from_struc(data, "lins_abs_sum_lp"); 

    end

end

if num_experiments > 1
    error("Multiple experiments not supported for the current plotting implementation, experiment data saved in 'save_folder' can be used for custom plotting")
    exit;
end

%% Comp: Convert Precision
qoi_analytical                      = convert_precision(qoi_analytical, "double");
qoi_hp                              = convert_precision(qoi_hp, "double");
qoi_lp                              = convert_precision(qoi_lp, "double");
qoi_model                           = convert_precision(qoi_model, "double");


%% Comp: Foward Bounds for the Linear System
lins_deterministic_e_fwd_bound      = comp_lins_fwd_bound_deterministic(lins_deterministic_e_bwd_bound, condition_lp);
lins_higham_e_fwd_bound             = comp_lins_fwd_bound_probabilistic(lins_higham_e_bwd_bound, condition_lp);
lins_bernstein_e_fwd_bound          = comp_lins_fwd_bound_probabilistic(lins_bernstein_e_bwd_bound, condition_lp);

%% Comp: Foward Bounds for the QoI 
qoi_deterministic_e_fwd_bound       = comp_qoi_fwd_bound_deterministic(eval_num_intervals, lower_precision, lins_deterministic_e_fwd_bound, lins_abs_sum_lp);
qoi_higham_e_fwd_bound              = comp_qoi_fwd_bound_higham(eval_num_intervals, lower_precision, lins_abs_sum_lp, condition_lp, prob_bound_confidence_levels);
qoi_bernstein_e_fwd_bound           = comp_qoi_fwd_bound_bernstein(eval_num_intervals, lower_precision, lins_abs_sum_lp, condition_lp, prob_bound_confidence_levels);

%% Comp: Mean Qoi
% Note: Sample mean_array_remains same for all the mean quantities
for ii = 1:length(eval_num_intervals)
    [qoi_analytical_mean(:, ii), sample_mean_array] = comp_sample_mean(qoi_analytical(:, ii), step_for_mean_convergence);
    qoi_hp_mean(:, ii)                              = comp_sample_mean(qoi_hp(:, ii), step_for_mean_convergence);
    qoi_lp_mean(:, ii)                              = comp_sample_mean(qoi_lp(:, ii), step_for_mean_convergence);
    qoi_model_mean(:, ii)                           = comp_sample_mean(qoi_model(:, ii), step_for_mean_convergence);
end

%% Comp: Mean Backward Error for the Linear System
for ii = 1:length(eval_num_intervals)
    lins_e_bwd_hp_mean(:, ii)                  = comp_sample_mean(lins_e_bwd_hp(:, ii), step_for_mean_convergence);
    lins_e_bwd_lp_mean(:, ii)                  = comp_sample_mean(lins_e_bwd_lp(:, ii), step_for_mean_convergence);
    lins_e_bwd_model_mean(:, ii)               = comp_sample_mean(lins_e_bwd_model(:, ii), step_for_mean_convergence);
end

%% Comp: Maximum Backward Error for the Linear System
lins_e_bwd_hp_max                   = max(lins_e_bwd_hp, [], 1);
lins_e_bwd_lp_max                   = max(lins_e_bwd_lp, [], 1);
lins_e_bwd_model_max                = max(lins_e_bwd_model, [], 1);

%% Comp: Mean Forward Error for the Qoi
for ii =1:length(eval_num_intervals)
    qoi_abs_error_true_mean(:, ii)             = comp_sample_mean(qoi_abs_error_true(:, ii), step_for_mean_convergence);
    qoi_abs_error_model_mean(:, ii)            = comp_sample_mean(qoi_abs_error_model(:, ii), step_for_mean_convergence);
    qoi_rel_error_true_mean(:, ii)             = comp_sample_mean(qoi_rel_error_true(:, ii), step_for_mean_convergence);
    qoi_rel_error_model_mean(:, ii)            = comp_sample_mean(qoi_rel_error_model(:, ii), step_for_mean_convergence);
end

%% Comp: Mean Forward Error Bounds for the QoI
for ii =1:length(eval_num_intervals)
    qoi_deterministic_e_fwd_bound_mean(:, ii)  = comp_sample_mean(qoi_deterministic_e_fwd_bound(:, ii), step_for_mean_convergence);
    for jj=1:length(prob_bound_confidence_levels)
        qoi_higham_e_fwd_bound_mean(:, ii, jj)         = comp_sample_mean(qoi_higham_e_fwd_bound(:, ii, jj), step_for_mean_convergence);
        qoi_bernstein_e_fwd_bound_mean(:, ii, jj)      = comp_sample_mean(qoi_bernstein_e_fwd_bound(:, ii, jj), step_for_mean_convergence);
    end
end


%%%%%%%%%%%%%%%%%
%%   Figures   %%
%%%%%%%%%%%%%%%%%
% Plot Backward Error for the Linear System
plot_lins_bwd_error(lins_e_bwd_lp, lins_deterministic_e_bwd_bound, lins_higham_e_bwd_bound, ...
                    lins_bernstein_e_bwd_bound, eval_num_intervals, lower_precision, prob_bound_confidence_levels, save_folder)

%% Plot Absolute Foward for the QOI
plot_qoi_abs_error_cdf(qoi_abs_error_true, qoi_abs_error_model, qoi_deterministic_e_fwd_bound, qoi_higham_e_fwd_bound, ...
                       qoi_bernstein_e_fwd_bound, eval_num_intervals, lower_precision, ...
                       prob_bound_confidence_levels, save_folder)


%% Plot Discretization Error
% plot_lins_discretization_error(true_discretization_error, discretization_max_error_bound, eval_num_intervals, lower_precision, save_folder)

%% Plot the Linear System Absolute Forward Error
discretization_data = load('./discretization_error_data.mat');
plot_lins_abs_fwd_error(lins_abs_error_true, lins_abs_error_model, discretization_data, lins_deterministic_e_fwd_bound, lins_higham_e_fwd_bound, ...
                        lins_bernstein_e_fwd_bound, eval_num_intervals, lower_precision, prob_bound_confidence_levels, save_folder)

%% Plot Mean Absolute Forward error vs. Samples for the QOI
% plot_qoi_mean_abs_error_vs_samples(qoi_abs_error_true_mean, qoi_abs_error_model_mean, qoi_deterministic_e_fwd_bound_mean, ...
%                                   qoi_higham_e_fwd_bound_mean, qoi_bernstein_e_fwd_bound_mean, eval_num_intervals, sample_mean_array, lower_precision, ...
%                                   prob_bound_confidence_levels, save_folder)

%% Plot the Solution Convergence with intervals
% plot_qoi_convergence_with_intervals(qoi_analytical_mean, qoi_hp_mean, qoi_lp_mean, qoi_model_mean, qoi_true_sol, eval_num_intervals, higher_precision, lower_precision, eval_parameter_samples, save_folder)

%% Plot the Solution Convergence with Samples
% plot_qoi_convergence_with_samples(qoi_analytical_mean, qoi_hp_mean, qoi_lp_mean, qoi_model_mean, qoi_true_sol, eval_num_intervals, higher_precision, lower_precision, eval_parameter_samples, sample_mean_array, save_folder)


%%%%%%%%%%%%%%%%%
%%  Functions  %%
%%%%%%%%%%%%%%%%%

function folder = get_save_folder()
    global save_folder;
    folder = save_folder;
end

function path = get_save_path(file_name)
    save_folder = get_save_folder();
    path        = strcat(save_folder, "/", file_name, ".mat");
end

function abs_error = comp_lins_abs_error(true_val, pred_val)
    true_val_p = convert_precision(true_val, "double");
    pred_val_p = convert_precision(pred_val, "double");
    abs_error  = norm( pred_val_p - true_val_p, inf );
end

function abs_sum  = comp_lins_abs_sum(sol)
    sol_p = convert_precision(sol, "double");
    abs_sum = sum(abs(sol_p));
end

function data = get_data_from_struc(struc, field_name)
    if isfield(struc, field_name) == false
        error("Field '%s' not found in the structure", field_name);
    else
        data = struc.(field_name);
    end
end

function lins_fwd_bound = comp_lins_fwd_bound_deterministic(lins_bwd_bound, condition)
    lins_fwd_bound = lins_bwd_bound .* condition;
    assert (isequal(size(lins_fwd_bound), size(condition)));
end

function lins_fwd_bound = comp_lins_fwd_bound_probabilistic(lins_bwd_bound, condition)
    num_eval_intervals      = size(lins_bwd_bound, 1);
    num_confidence_levels   = size(lins_bwd_bound, 2);
    num_parameter_samples   = size(condition, 1);
    lins_fwd_bound = zeros(num_parameter_samples, num_eval_intervals, num_confidence_levels); 
    for ii = 1:num_eval_intervals
        for jj = 1:num_confidence_levels
            lins_fwd_bound(:, ii, jj) = lins_bwd_bound(ii, jj) .* condition(:, ii);
        end
    end

    assert (size(lins_fwd_bound, 1) == num_parameter_samples);
    assert (size(lins_fwd_bound, 2) == num_eval_intervals);
    assert (size(lins_fwd_bound, 3) == num_confidence_levels);
end

function error_norm = comp_true_numerical_discretization_error(lins_anal, lins_hp)
    error = lins_anal - lins_hp;
    error_norm = norm(error, inf);
end