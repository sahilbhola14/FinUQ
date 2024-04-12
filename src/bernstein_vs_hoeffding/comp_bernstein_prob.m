% Function computes the product of rounding error bounding using bernstein concentration inequality
% Rounding error distribution is assumed to be Uniform(- unit_round_off, unit_round_off)

function prob = comp_bernstein_prob(prob_size, precision, lambda_vals)
    % prob_size: n, size of the problem
    % precision: p, precision of the floating point
    % lambda_vals: vector of lambda values to compute the probability for.

    % Unit round off
    urd         = comp_urd(precision);

    % Variance of 'n' independent rounding errors
    variance    = comp_variance_log_one_rounding_error(precision);
    variance_n  = variance * prob_size;

    % Rounding error bound using  Bersntein concentration inequality
    t           = lambda_vals.*sqrt(prob_size)*urd;
    numerator   = -t.^2;
    denominator = 2 * ( variance_n + ((lambda_vals * sqrt(prob_size) * urd^2) / (3 * (1 - urd))) ); 
    prob        = 1 - 2 * exp(numerator./denominator);

end