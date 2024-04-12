function qoi_fwd_bound = comp_qoi_fwd_bound_deterministic(eval_num_intervals, precision, lins_e_fwd_bound, lins_abs_sum)
    num_samples = size(lins_e_fwd_bound, 1);

    qoi_fwd_bound = zeros(num_samples, length(eval_num_intervals)); 

    for ii = 1:length(eval_num_intervals)
        delta_x     = 1 / eval_num_intervals(ii);
        system_size = (1-delta_x)/delta_x;

        gamma_n = compute_determistic_gamma_n(system_size, precision);
        qoi_fwd_bound(:, ii) = delta_x * (system_size * lins_e_fwd_bound(:, ii) + gamma_n * lins_abs_sum(:, ii));
    end

    assert (size(qoi_fwd_bound, 1) == size(lins_abs_sum, 1));
    assert (size(qoi_fwd_bound, 2) == length(eval_num_intervals));
end

%% Funcitons
function gamma_n = compute_determistic_gamma_n(n, precision)
    urd     = comp_urd(precision);
    gamma_n = (n*urd) / (1 - (n*urd));
end