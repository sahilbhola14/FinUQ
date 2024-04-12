function qoi_fwd_bound = comp_qoi_fwd_bound_higham(eval_num_intervals, precision, lins_abs_sum, condition, confidence_levels)
    num_samples             = size(lins_abs_sum, 1);
    num_confidence_levels   = length(confidence_levels);

    qoi_fwd_bound           = zeros(num_samples, length(eval_num_intervals), num_confidence_levels); 

    for ii = 1:length(eval_num_intervals)
        delta_x                 = 1 / eval_num_intervals(ii);
        system_size             = (1-delta_x)/delta_x;
        qoi_fwd_bound(:, ii, :) = comp_bound_per_interval(system_size, precision, confidence_levels, condition(:, ii), lins_abs_sum(:, ii), delta_x);
    end

    assert (size(qoi_fwd_bound, 1) == num_samples);
    assert (size(qoi_fwd_bound, 2) == length(eval_num_intervals));
    assert (size(qoi_fwd_bound, 3) == num_confidence_levels);
end

%% Functions
function qoi_fwd_bound = comp_bound_per_interval(system_size, precision, confidence_levels, condition, lins_abs_sum, delta_x)
    % Test Lambdas
    test_lambdas = logspace(0, 2, 200000);
    % Probability of the ODE solution
    T_ODE        = comp_T_ODE(system_size, precision, test_lambdas);
    % Lambda critical
    for ii = 1:length(confidence_levels)
        lambda_critical     = comp_lambda_critical(T_ODE, test_lambdas, confidence_levels(ii));

        % Foward Bound for thomas with ODE probability for critical lambda
        gamma_one_critical  = comp_probabilistic_gamma_n(1, precision, lambda_critical);
        gamma_two_critical  = comp_probabilistic_gamma_n(2, precision, lambda_critical);
        e_bwd_thomas = 2*gamma_one_critical + gamma_two_critical + gamma_one_critical*gamma_two_critical;
        e_fwd_thomas = e_bwd_thomas * condition;

        % Forward Bound for the Complete Solution
        gamma_n = comp_probabilistic_gamma_n(system_size, precision, lambda_critical);
        qoi_fwd_bound(:, ii) = delta_x * (system_size * e_fwd_thomas + gamma_n * lins_abs_sum);
    end

    assert (size(qoi_fwd_bound, 1) == size(lins_abs_sum, 1));
    assert (size(qoi_fwd_bound, 2) == length(confidence_levels));
   
end

function T_ODE = comp_T_ODE(system_size, precision, test_lambdas)
    % Thomas Probability
    T_LS = comp_thomas_prob_higham(system_size, precision, test_lambdas);
    % Summation Probability
    T_D = comp_summation_probability(system_size, precision, test_lambdas);
    % ODE Probability
    T_ODE = 1 - ( (1-T_LS) + (1-T_D) );
end

function prob = comp_summation_probability(system_size, precision, test_lambdas)
    p_lambda = comp_hoeffding_prob(precision, test_lambdas);
    % Summation probability
    q_lambda_sum = 1 - system_size*(1 - p_lambda);
    % Total probability
    prob = q_lambda_sum;
end

function lambda_critical = comp_lambda_critical(prob, lambda_vals, confidence_levels)
    lambda_critical = zeros(1, length(confidence_levels));
    for ii = 1:length(confidence_levels)
        confidence_level = confidence_levels(ii);
        lambda_critical(ii) = lambda_vals(find(prob >= confidence_level, 1, 'first'));
    end
    assert (length(lambda_critical) == length(confidence_levels));
end

function gamma_n = comp_probabilistic_gamma_n(n, precision, lambda_vals)
    urd     = comp_urd(precision);
    gamma_n = lambda_vals.*sqrt(n)*urd;

    assert (length(gamma_n) == length(lambda_vals));
end