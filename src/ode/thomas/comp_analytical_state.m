function anal_u = comp_analytical_state(parameters, M)

    x = linspace(0, 1, M+1);
    x = x(2:end-1);

    theta_1     = convert_precision(parameters(:, 1), "double");
    theta_2     = convert_precision(parameters(:, 2), "double");

    numerator   = -50 * theta_2.^2 * (x*log(1 + theta_1) - log(1 + theta_1*x));
    denominator = theta_1 * log(1+theta_1);
    anal_u = numerator ./ denominator;
    assert (length(anal_u) == length(x), 'Invalid length of analytical solution.');
    anal_u = anal_u.';
end