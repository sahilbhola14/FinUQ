function lins_bwd_bound = comp_lins_bwd_bound_higham(eval_num_intervals, precision, confidence_levels)
    % eval_num_intervals: number of intervals of the ODE system
    % precision: precision of the computation
    % confidence_levels: the confidence levels to be used in the computation of the bound

    % Test Lambdas
    test_lambdas = get_test_lambda();

    lins_bwd_bound = zeros(length(eval_num_intervals), length(confidence_levels));

    for ii = 1:length(eval_num_intervals)
        num_intervals = eval_num_intervals(ii);

        % System size
        system_size = num_intervals - 1;

        % Compute the higham probability for the test lambdas
        T_LS = comp_thomas_prob_higham(system_size, precision, test_lambdas);

        % Find Lambda critical 
        lambda_critical = comp_lambda_critical(T_LS, test_lambdas, confidence_levels);

        % Compute Linear system bwd bound
        gamma_one_critical = comp_probabilistic_gamma_n(1, precision, lambda_critical);
        gamma_two_critical = comp_probabilistic_gamma_n(2, precision, lambda_critical);
        lins_bwd_bound(ii, :) = 2.0*gamma_one_critical + gamma_two_critical + gamma_one_critical.*gamma_two_critical;

    end
    assert (size(lins_bwd_bound, 1) == length(eval_num_intervals));
    assert (size(lins_bwd_bound, 2) == length(confidence_levels));

end

%% Supplementary functions
function gamma_n = comp_probabilistic_gamma_n(n, precision, lambda_vals)
    urd     = comp_urd(precision);
    gamma_n = lambda_vals.*sqrt(n)*urd;

    assert (length(gamma_n) == length(lambda_vals));
end