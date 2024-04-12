function gamma = dot_product_bwd_bound_deterministic(vector_size, precision)
    gamma = compute_determistic_gamma_n(vector_size, precision);
end

function gamma_n = compute_determistic_gamma_n(n, precision)
    urd     = comp_urd(precision);
    gamma_n = (n*urd) / (1 - (n*urd));
end