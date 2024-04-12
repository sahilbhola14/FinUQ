function anal_p = comp_analytical_integral(parameters)

    theta_1     = convert_precision(parameters(:, 1), "double");
    theta_2     = convert_precision(parameters(:, 2), "double");

    numerator   = (25.0*theta_2.^2).*(-2*theta_1 + (2 + theta_1).*log(1 + theta_1));
    denominator = (theta_1.^2).*log(1 + theta_1);
    anal_p      = numerator./denominator;

end
