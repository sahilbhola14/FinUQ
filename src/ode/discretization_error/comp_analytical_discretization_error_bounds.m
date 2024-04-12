function [e_sup, e_inf, e_max] = comp_analytical_discretization_error_bounds(parameters, num_intervals)

    assert (size(parameters, 1) == 1 && size(parameters, 2) == 2)

    delta_x         = 1 / num_intervals;
    grid            = delta_x:delta_x:(num_intervals-1)*delta_x;

    % Compute the analytical solution
    u_anal_discrete = compute_analytical_solution(grid, parameters);

    % Compute the diagonal coefficients
    [sub_diag, main_diag, super_diag] = get_diagonal(parameters, num_intervals);

    % Compute the A matrix
    A_mat = construct_A_mat_from_diags(sub_diag, main_diag, super_diag);

    % Compute t_sup and t_inf
    [t_sup, t_inf] = compute_t_sup_t_inf(parameters, [sub_diag; main_diag; super_diag], num_intervals, u_anal_discrete);

    % Error 
    inv_A = inv(A_mat);
    e_sup = inv_A * t_sup';
    e_inf = inv_A * t_inf';

    % Maximum
    e_max = norm(max(abs(e_sup), abs(e_inf)), inf);

end

%% Functions
function A = construct_A_mat_from_diags(sub_diag, main_diag, super_diag)
    A = diag(sub_diag(2:end), -1) + diag(main_diag, 0) + diag(super_diag(1:end-1), 1);
end

function parameters = sample_parameters(num_samples)
    theta_1 = rand(num_samples, 1) + 0.1;
    theta_2 = randn(num_samples, 1);
    parameters = [theta_1, theta_2];
end

function u_analytical = compute_analytical_solution(input_space, parameters)
    [theta_1, theta_2] = get_parameters(parameters);
    numerator   = -(50*theta_2^2)*(input_space.*(log(1 + theta_1)) - log(1 + theta_1*input_space));
    denominator = theta_1*log(1 + theta_1);
    u_analytical = numerator./denominator;
end

function first_derivative = compute_first_derivative(input, parameters)
    % Compute the first derivative of the analytical solution
    [theta_1, theta_2] = get_parameters(parameters);
    numerator = (50*theta_2^2).*(-(theta_1 ./ (1 + theta_1.*input)) + log(1 + theta_1));
    denominator = theta_1*log(1 + theta_1);
    first_derivative = -numerator./denominator;
end

function second_derivative = compute_second_derivative(input, parameters)
    % Compute the second derivative of the analytical solution
    [theta_1, theta_2] = get_parameters(parameters);
    numerator = 50*theta_1*theta_2^2;
    denominator = ((1 + theta_1.*input).^2) * log(1 + theta_1);
    second_derivative = -numerator./denominator;
end

function third_derivative = compute_third_derivative(input, parameters)
    % Compute the third derivative of the analytical solution
    [theta_1, theta_2] = get_parameters(parameters);
    numerator = 100*theta_1^2*theta_2^2;
    denominator = ((1 + theta_1.*input).^3) * log(1 + theta_1);
    third_derivative = numerator./denominator;
end


function [theta_1, theta_2] = get_parameters(parameters)
    theta_1 = parameters(1);
    theta_2 = parameters(2);
end

function [sub_diag, main_diag, super_diag] = get_diagonal(parameters, num_intervals)
    % Function computes the diagonals (Interior points)
    delta_x = 1 / num_intervals;
    delta_x_sq = delta_x * delta_x;
    index = 1:num_intervals-1;
    [theta_1, theta_2] = get_parameters(parameters);
    sub_diag = (1 / delta_x_sq) + (theta_1/delta_x)*(index - 0.5);
    super_diag = (1 / delta_x_sq) + (theta_1/delta_x)*(index + 0.5);
    main_diag = -(2 / delta_x_sq) - 2*(theta_1/delta_x)*(index);
end

function [t_sup, t_inf] = compute_t_sup_t_inf(parameters, diagonals, num_intervals, u_anal_discrete)
    % Get the grid
    delta_x         = 1 / num_intervals; 
    delta_x_sq      = delta_x * delta_x;
    delta_x_cu      = delta_x_sq * delta_x;
    grid            = delta_x:delta_x:(num_intervals-1)*delta_x;

    % Function computes the t_sup and t_inf
    [theta_1, theta_2] = get_parameters(parameters);
    sub_diag    = diagonals(1, :);
    main_diag   = diagonals(2, :);
    super_diag  = diagonals(3, :);

    % Constant term
    alpha           = zeros(1, num_intervals-1);
    beta            = zeros(1, num_intervals-1);
    gamma           = zeros(1, num_intervals-1);
    alpha(2:end)    = sub_diag(2:end);
    beta            = main_diag;
    gamma(1:end-1)  = super_diag(1:end-1);

    % term_1
    term_1                      = (alpha + beta + gamma) .* u_anal_discrete;

    % term_2
    first_derivative            = compute_first_derivative(grid, parameters);
    term_2                      = delta_x * (alpha - gamma).*first_derivative;

    % term_3
    second_derivative           = compute_second_derivative(grid, parameters);
    term_3                      = (delta_x_sq/2) * (alpha + gamma).*second_derivative;

    % term_4 (Sup)
    third_derivative            = compute_third_derivative(grid, parameters);
    term_4_sup                  = (delta_x_cu/6) * (alpha.*third_derivative - gamma.*third_derivative);

    % term_4 (Inf)
    grid_plus_delta_x           = grid + delta_x;
    grid_minus_delta_x          = grid - delta_x;
    third_derivative_grid_plus  = compute_third_derivative(grid_plus_delta_x, parameters);
    third_derivative_grid_minus = compute_third_derivative(grid_minus_delta_x, parameters);
    term_4_inf                  = (delta_x_cu/6) * (alpha.*third_derivative_grid_plus - gamma.*third_derivative_grid_minus);

    % t_sup 
    t_sup = term_1 + term_2 + term_3 + term_4_sup + 50*theta_2^2;

    % t_inf
    t_inf = term_1 + term_2 + term_3 + term_4_inf + 50*theta_2^2;
end