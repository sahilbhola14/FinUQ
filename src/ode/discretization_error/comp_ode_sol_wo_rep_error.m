function [p_sol, u_sol, e_bwd_lin_sys, e_fwd_lin_sys, approx_condition] = comp_ode_sol_wo_rep_error(num_intervals, parameters, solve_precision, lower_precision, verbose, checks)
    % function computes the ode solution w/o representation error (lower precsion parameters are used)
    % Args: num_intervals - number of intervals
    %       parameters - parameters of the model (theta_1, theta_2)
    %       solve_precision - precision of the ode solver (System is solved in this precision)
    %       lower_precision - precision of the ode solver (Lower precion for enforcing no representation error)

    % Note: In order to change precision the parameteters and grid size are changed to
    % a priori. For some reason if the generated diagonals are converted to the desired
    % precision, the half precision does not work for finer grids. 

    % Note: Lhs and Rhs of the system are modified a bit to avoid overflow and underflow

    % Note: No representation error is introduced in the computation of the solution
    % The parapeters and the vectos of the system are obtained in {lower_precision} they 
    % are converted to {solve_precision}. This ensures that there is no representation error

    % Returns: p_sol - integrated solution
    %          u_sol - solution of the linear system
    %          e_bwd_lin_sys - backward error of the linear system
    %          e_fwd_lin_sys - forward error of the linear system
    %          approx_condition - approximate condition number of the linear system (forward error \approx backward error * approx_condition)

    % Header
    if verbose
        print_header(num_intervals, parameters, solve_precision, lower_precision);
    end

    % Grid size
    delta_x = 1 / num_intervals;

    % Grid
    grid    = delta_x: delta_x : (num_intervals - 1) * delta_x;

    % Construct Diagonals (in {lower_precision})
    [sub_diag, main_diag, super_diag]   = get_diagonals(parameters, num_intervals, lower_precision, checks);

    % Construct the rhs (in {lower_precision})
    rhs                                 = get_rhs(parameters, num_intervals, lower_precision, checks);

    % Convert the diagonals to {solve_precision}
    sub_diag_p                          = convert_precision(sub_diag, solve_precision);
    main_diag_p                         = convert_precision(main_diag, solve_precision);
    super_diag_p                        = convert_precision(super_diag, solve_precision);

    % Convert the rhs to {solve_precision}
    rhs_p                               = convert_precision(rhs, solve_precision);

    % Compute the Decomposition (Converted precision vectors are used to construct the decomposition)
    [a_array, b_array]                  = compute_decomposition(sub_diag_p, main_diag_p, super_diag_p, solve_precision, checks);

    % Forward Substitution (Converted precision vectors are used to construct the solution)
    y_sol                               = forward_substitution(a_array, rhs_p, solve_precision, checks);

    % Backward Substitution (Converted precision vectors are used to construct the solution)
    u_sol                               = backward_substitution(b_array, y_sol, super_diag_p, solve_precision, checks);

    if all (u_sol == 0) == 1
        fprintf('u_sol\n');
        u_sol
    end
    assert (all(u_sol == 0) == 0, "Solution is zero");

    % Backward Error for the solution of the linear system
    e_bwd_lin_sys                       = compute_e_bwd_lin_sys(sub_diag_p, main_diag_p, super_diag_p, a_array, b_array, u_sol, rhs_p, checks);

    % Foward Error for the solution of the linear system
    [e_fwd_lin_sys, approx_condition]   = compute_e_fwd_lin_sys(sub_diag_p, main_diag_p, super_diag_p, a_array, b_array, u_sol, rhs_p, checks, e_bwd_lin_sys);

    % Integrate the Solution
    p_sol                               = integrate_solution(u_sol, delta_x, solve_precision, checks);

end

%% Supporting Functions
function [sub_diag, main_diag, super_diag] = get_diagonals(parameters, num_intervals, precision, checks)

    delta_x     = 1 / num_intervals;
    delta_x_sq  = delta_x * delta_x;

    % Get the parameters
    [theta_1, theta_2] = get_parameters(parameters);

    % Convert precision
    delta_x     = convert_precision(delta_x, precision);
    delta_x_sq  = convert_precision(delta_x_sq, precision);
    theta_1     = convert_precision(theta_1, precision);
    theta_2     = convert_precision(theta_2, precision);
    one_half    = convert_precision(0.5, precision);
    one         = convert_precision(1, precision);   

    % Constant Vector (Shared for all diagonals)
    idx             = 1:num_intervals;
    x_array         = idx * delta_x;
    % constant_vector = (one / delta_x) + theta_1 * (idx - one_half);
    constant_vector = one + theta_1 * (x_array - one_half*delta_x);

    % Sub Diagonal
    sub_diag        = constant_vector(1:num_intervals - 1);

    % Super Diagonal
    super_diag      = constant_vector(2:num_intervals);

    % Main Diagonal
    main_diag       = -1 * (sub_diag + super_diag);

    % Checks
    if checks
        assert(isa(sub_diag, precision));
        assert(isa(main_diag, precision));
        assert(isa(super_diag, precision));
        check_inf_nan(sub_diag, "sub_diag");
        check_inf_nan(main_diag, "main_diag");
        check_inf_nan(super_diag, "super_diag");
    end

end

function [theta_1, theta_2] = get_parameters(parameters)
    theta_1 = parameters(1);
    theta_2 = parameters(2);
end

function rhs = get_rhs(parameters, num_intervals, precision, checks)
    delta_x = 1 / num_intervals;
    delta_x = convert_precision(delta_x, precision);

    [theta_1, theta_2] = get_parameters(parameters);
    % Convert precision
    theta_1 = convert_precision(theta_1, precision);
    theta_2 = convert_precision(theta_2, precision);

    rhs     = -50*theta_2*theta_2* ones(num_intervals - 1, 1) * delta_x * delta_x;

    % Checks
    if checks
        assert(isa(rhs, precision));
        check_inf_nan(rhs, "rhs");
    end
end

function [a_array, b_array] = compute_decomposition(sub_diag, main_diag, super_diag, precision, checks)
    % Function computes the decomposition of the tridiagonal matrix
    % Args: sub_diag - sub diagonal of the tridiagonal matrix
    %       main_diag - main diagonal of the tridiagonal matrix
    %       super_diag - super diagonal of the tridiagonal matrix
    % Returns: a_array - array for the Lower triangular matrix (Sub Diagonal)
    %          b_array - array for the Upper triangular matrix (Diagonal)
    % Get the size of the matrix
    n       = length(main_diag);
    % Initialize the arrays
    a_array = convert_precision(zeros(n, 1), precision);
    b_array = convert_precision(zeros(n, 1), precision);

    b_array(1) = main_diag(1);
    for ii = 2:n
      a_array(ii) = sub_diag(ii) / b_array(ii - 1);  
      b_array(ii) = main_diag(ii) - a_array(ii) * super_diag(ii - 1);
    end

    % Checks
    if checks
        assert(isa(a_array, precision));
        assert(isa(b_array, precision));
        check_inf_nan(a_array, "a_array");
        check_inf_nan(b_array, "b_array");
    end
end

function y_sol = forward_substitution(a_array, rhs, precision, checks)
    y_sol = convert_precision(zeros(length(rhs), 1), precision);

    y_sol(1) = rhs(1);
    for ii = 2:length(rhs)
        y_sol(ii) = rhs(ii) - a_array(ii) * y_sol(ii - 1);
    end

    % Checks
    if checks
        assert(isa(y_sol, precision));
        check_inf_nan(y_sol, "y_sol");
    end
end

function u_sol = backward_substitution(b_array, y_sol, super_diag, precision, checks)
    u_sol = convert_precision(zeros(length(y_sol), 1), precision);

    u_sol(end) = y_sol(end) / b_array(end);
    for ii = length(y_sol) - 1:-1:1
        u_sol(ii) = (y_sol(ii)  - super_diag(ii) * u_sol(ii + 1)) / b_array(ii);
    end

    % Checks
    if checks
        assert (isa(u_sol, precision));
        check_inf_nan(u_sol, "u_sol");
    end
end

function p_sol = integrate_solution(u_sol, delta_x, precision, checks)
    p_sol = convert_precision(0, precision);

    for ii = 1:length(u_sol)
        p_sol = p_sol + u_sol(ii);
    end
    p_sol = p_sol * delta_x;

    % Checks
    if checks
        assert (isa(p_sol, precision));
        check_inf_nan(p_sol, "p_sol")
    end
end

function check_inf_nan(val, name)
    check = sum(isinf(val)) + sum(isnan(val));

    if check > 0
        val
    end
    assert (check == 0, "%s has inf or nan values", name);
end

function print_header(num_intervals, parameters, solve_precision, lower_precision)
    fprintf("Computing ODE Solution with %d intervals, parameters = [%f, %f], solve_precision = %s, lower_precision = %s\n", num_intervals, parameters(1), parameters(2), solve_precision, lower_precision);
end

function A = construct_A_mat_from_diags(sub_diag, main_diag, super_diag)
    A = diag(sub_diag(2:end), -1) + diag(main_diag, 0) + diag(super_diag(1:end-1), 1);
end

function [L, U] = construct_LU_mat_from_arrays(a_array, b_array, super_diag)
    n = length(b_array);
    L = eye(n) + diag(a_array(2:end), -1);
    U = diag(b_array, 0) + diag(super_diag(1:end-1), 1);
end

function e_bwd = compute_e_bwd_lin_sys(sub_diag, main_diag, super_diag, a_array, b_array, u_sol, rhs, checks)
    % Convert precision to Double
    sub_diag_p      = convert_precision(sub_diag, "double");
    main_diag_p     = convert_precision(main_diag, "double");
    super_diag_p    = convert_precision(super_diag, "double");
    a_array_p       = convert_precision(a_array, "double");
    b_array_p       = convert_precision(b_array, "double");
    u_sol_p         = convert_precision(u_sol, "double");
    rhs_p           = convert_precision(rhs, "double");

    % Construct A matrix
    A               = construct_A_mat_from_diags(sub_diag_p, main_diag_p, super_diag_p);

    % Construct L and U matrix from a_array, b_array, and super_diag
    [L, U]          = construct_LU_mat_from_arrays(a_array_p, b_array_p, super_diag_p);

    % Numerator
    numerator       = abs(rhs_p - A * u_sol_p);
    % Denominator
    denominator     = abs(L) * abs(U) * abs(u_sol_p);
    % Backward Error
    if checks
        check_inf_nan(numerator, "numerator");
        if all(denominator ~= 0) == 0
            fprintf('L\n');
            L
            fprintf('U\n');
            U
            fprintf('u_sol_p\n');
            u_sol_p
        end
        assert (all(denominator ~= 0), "Denominator is zero")
        check_inf_nan(denominator, "denominator");
    end
    e_bwd           = norm(numerator ./ denominator, inf);

    % Checks
    if checks
        assert(isa(e_bwd, "double"));
        check_inf_nan(e_bwd, "e_bwd");
    end
end

function [e_fwd_lin_sys, approx_condition]   = compute_e_fwd_lin_sys(sub_diag, main_diag, super_diag, a_array, b_array, u_sol, rhs, checks, e_bwd_lin_sys)
    % Convert precision to Double
    sub_diag_p      = convert_precision(sub_diag, "double");
    main_diag_p     = convert_precision(main_diag, "double");
    super_diag_p    = convert_precision(super_diag, "double");
    a_array_p       = convert_precision(a_array, "double");
    b_array_p       = convert_precision(b_array, "double");
    u_sol_p         = convert_precision(u_sol, "double");
    rhs_p           = convert_precision(rhs, "double");
    e_bwd_lin_sys_p = convert_precision(e_bwd_lin_sys, "double");

    % Construct A matrix
    A               = construct_A_mat_from_diags(sub_diag_p, main_diag_p, super_diag_p);

    % Construct L and U matrix from a_array, b_array, and super_diag
    [L, U]          = construct_LU_mat_from_arrays(a_array_p, b_array_p, super_diag_p);

    % Inverse of A matrix
    assert (det(A)  ~= 0, "A is singular");
    inv_A           = inv(A);

    % Conditioning
    approx_condition    = norm(abs(inv_A) * abs(L) * abs(U) * abs(u_sol_p), inf);

    % Forward Error
    e_fwd_lin_sys       = e_bwd_lin_sys_p * approx_condition;

    if checks
        assert(isa(e_fwd_lin_sys, "double"));
        assert(isa(approx_condition, "double"));
        check_inf_nan(e_fwd_lin_sys, "e_fwd_lin_sys");
        check_inf_nan(approx_condition, "approx_condition");
    end
end