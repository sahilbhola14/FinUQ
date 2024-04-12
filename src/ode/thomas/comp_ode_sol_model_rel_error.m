function [p_sol, u_sol, e_bwd_lin_sys] = comp_ode_sol_model_rel_error(num_intervals, parameters, solve_precision, lower_precision, verbose, checks)
    % function computes the ode solution w/o representation error (lower precsion parameters are used)
    % Args: num_intervals - number of intervals
    %       parameters - parameters of the model (theta_1, theta_2)
    %       solve_precision - precision of the ode solver (System is solved in this precision)
    %       lower_precision - precision of the ode solver (Lower precision for enforcing no representation error and sampling the relative errors)

    % Note: In order to change precision the parameteters and grid size are changed to
    % a priori. For some reason if the generated diagonals are converted to the desired
    % precision, the half precision does not work for finer grids. 

    % Note: Lhs and Rhs of the system are modified a bit to avoid overflow and underflow

    % Note: To model the {solve_precision} we use the arithmetic model fl(a op b)  = (a op b)(1 + delta)
    % where we sample the delta from a uniform distribution in [-u, u] where u is the unit round off

    % Checks
    assert (solve_precision == "double", "For relative error overload, solve precision is ALWAYS double");

    % Define global variables
    global lower_sample_precision;
    lower_sample_precision = lower_precision;

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

    % Backward Error for the solution of the linear system
    e_bwd_lin_sys                       = compute_e_bwd_lin_sys(sub_diag_p, main_diag_p, super_diag_p, a_array, b_array, u_sol, rhs_p, checks);

    % Integrate the Solution
    p_sol                               = integrate_solution(u_sol, delta_x, solve_precision, checks);
end

%% Supporting Functions
function [sub_diag, main_diag, super_diag] = get_diagonals(parameters, num_intervals, precision, checks)

    delta_x = 1 / num_intervals;
    delta_x_sq = delta_x * delta_x;

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
    sub_diag = constant_vector(1:num_intervals - 1);

    % Super Diagonal
    super_diag = constant_vector(2:num_intervals);

    % Main Diagonal
    main_diag = -1 * (sub_diag + super_diag);

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

    [theta_1, theta_2] = get_parameters(parameters);
    % Convert precision
    theta_1 = convert_precision(theta_1, precision);
    theta_2 = convert_precision(theta_2, precision);
    rhs = -50*theta_2*theta_2* ones(num_intervals - 1, 1) * delta_x * delta_x;

    % Checks
    if checks
        assert(isa(rhs, precision));
        check_inf_nan(rhs, "rhs");
    end

end

function [a_array, b_array] = compute_decomposition(sub_diag, main_diag, super_diag, precision, checks)
    % Function computes the decomposition of the tridiagonal matrix (Doolittle's Decomposition)
    % Args: sub_diag - sub diagonal of the tridiagonal matrix
    %       main_diag - main diagonal of the tridiagonal matrix
    %       super_diag - super diagonal of the tridiagonal matrix
    % Returns: a_array - array for the Lower triangular matrix (Sub Diagonal)
    %          b_array - array for the Upper triangular matrix (Diagonal)
    % Get the size of the matrix
    n = length(main_diag);
    % Initialize the arrays
    a_array = convert_precision(zeros(n, 1), precision);
    b_array = convert_precision(zeros(n, 1), precision);
    b_array(1) = main_diag(1);
    for ii = 2:n
        ops_pert = get_rel_error_perturbation(1);
        a_array(ii) = sub_diag(ii) / (b_array(ii - 1) * ops_pert(1));  

        ops_pert = get_rel_error_perturbation(2);
        b_array(ii) = (main_diag(ii) - (a_array(ii) * super_diag(ii - 1) * ops_pert(1))) / ops_pert(2);
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
    assert(precision == "double")
    y_sol = convert_precision(zeros(length(rhs), 1), precision);
    y_sol(1) = rhs(1);
    for ii = 2:length(rhs)
        ops_pert  = get_rel_error_perturbation(2);
        y_sol(ii) = (rhs(ii) - (a_array(ii) * y_sol(ii - 1) * ops_pert(1))) / ops_pert(2);
    end

    % Checks
    if checks
        assert(isa(y_sol, precision));
        check_inf_nan(y_sol, "y_sol");
    end
end

function u_sol = backward_substitution(b_array, y_sol, super_diag, precision, checks)
    assert(precision == "double");
    u_sol = convert_precision(zeros(length(y_sol), 1), precision);
    ops_pert   = get_rel_error_perturbation(1);
    u_sol(end) = y_sol(end) / (b_array(end) * ops_pert(1));
    for ii = length(y_sol) - 1:-1:1
        ops_pert  = get_rel_error_perturbation(3);
        u_sol(ii) = (y_sol(ii)  - (super_diag(ii) * u_sol(ii + 1) * ops_pert(1)) ) / (b_array(ii) * ops_pert(2) * ops_pert(3));
    end

    % Checks
    if checks
        assert (isa(u_sol, precision));
        check_inf_nan(u_sol, "u_sol");
    end
end

function p_sol = integrate_solution(u_sol, delta_x, precision, checks)
    assert(precision == "double")
    p_sol = convert_precision(0, precision);
    for ii = 1:length(u_sol)
        ops_pert = get_rel_error_perturbation(1);
        p_sol = (p_sol + u_sol(ii)) * ops_pert(1);
    end
    ops_pert = get_rel_error_perturbation(1);
    p_sol = p_sol * delta_x * ops_pert(1);

    % Checks
    if checks
        assert (isa(p_sol, precision));
        check_inf_nan(p_sol, "p_sol")
    end
end

function val_converted = convert_precision(val, precision)
    if precision == "double"
        val_converted = double(val);
    elseif precision == "single"
        val_converted = single(val);
    elseif precision == "half"
        val_converted = half(val);
    else
        error("Invalid Precision");
    end
end

function check_inf_nan(val, name)
    check = sum(isinf(val)) + sum(isnan(val));
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
    e_bwd           = norm(numerator / denominator, inf);

    % Checks
    if checks
        assert(isa(e_bwd, "double"));
        check_inf_nan(e_bwd, "e_bwd");
    end
end

function lower_precision = get_lower_precision()
    global lower_sample_precision;
    lower_precision = lower_sample_precision;
end

function urd = get_unit_round_off(precision)
    if precision == "double"
        meps = convert_precision(eps("double"), "double");
    elseif precision == "single"
        meps = convert_precision(eps("single"), "double");
    elseif precision == "half"
        meps = convert_precision(eps("half"), "double");
    else
        error("Invalid Precision");
    end

    urd = meps / 2.0;
end

function samples = get_uniform_samples(num_samples)
    samples = convert_precision(rand(num_samples, 1), "double");
end

function perturbation = get_rel_error_perturbation(num_samples)
    precision                       = get_lower_precision();
    [rel_error_samp , perturbation] = sample_rel_error(num_samples, precision);
end