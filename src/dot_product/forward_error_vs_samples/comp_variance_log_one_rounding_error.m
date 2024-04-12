function variance = comp_variance_log_one_rounding_error(precision)
    % Computes the variance of log(1+\delta) if \delta is uniformly distributed in [-unit_round_off,unit_round_off]
    % Unit round off
    urd         = comp_urd(precision);
    % variance 
    kappa       = (-1 + urd^2);
    numerator   = (4*urd^2) + kappa * (log(1-urd)^2 - 2*log(1-urd)*log(1+urd) + log(1+urd)^2);
    denominator = 4*urd^2;
    variance    = numerator/denominator;
end