% Function computes the product of rounding error bounding using hoeffidng (Higham and Mary, 2019)
function prob = comp_hoeffding_prob(precision, lambda_vals)
    % precision: p, precision of the floating point
    % lambda_vals: vector of lambda values to compute the probability for.

    % Unit round off
    urd = comp_urd(precision);


    % Rounding error bound using Hoeffding concentration inequality
    prob = 1 - 2*exp(-0.5*(lambda_vals.*(1-urd)).^2);

    assert (length(prob) == length(lambda_vals));

end


