function [gamma, lambda_c, T_D_out] = dot_product_bwd_bound_bernstein(vector_size, precision, confidence_levels)
    % vector_size: the size of the vector (or the problem size)
    % precision: precision of the computation
    % confidence_levels: the confidence levels to be used in the computation of the bound

    % compuet urd
    urd = comp_urd(precision);

    % Test Lambdas
    test_lambdas = get_test_lambda();

    % Compute the probability bound for the test lambdas
    T_D = zeros(1, length(test_lambdas));
    parfor ii = 1:vector_size
        size_i = vector_size - max(2, ii) + 2;
        p_b_i  = comp_bernstein_prob(size_i, precision, test_lambdas);
        T_D    = T_D + (1 - p_b_i);
    end
    T_D = 1 - T_D;

    % Find Lambda critical
    lambda_critical = comp_lambda_critical(T_D, test_lambdas, confidence_levels);

    % Compute gamma
    gamma = lambda_critical.*sqrt(vector_size)*urd;

    if nargout > 1
        lambda_c = lambda_critical;
        T_D_out = T_D;
    end
end