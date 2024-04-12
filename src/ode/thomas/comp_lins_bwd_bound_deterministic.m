function lins_bwd_bound = comp_lins_bwd_bound_deterministic(precision)
    gamma_one = compute_determistic_gamma_n(1, precision);
    gamma_two = compute_determistic_gamma_n(2, precision);
    lins_bwd_bound = 2*gamma_one  + gamma_two + gamma_one* gamma_two;
end

function gamma_n = compute_determistic_gamma_n(n, precision)
    urd     = comp_urd(precision);
    gamma_n = (n*urd) / (1 - (n*urd));
end