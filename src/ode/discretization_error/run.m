void;
eval_num_samples = 10000;
eval_num_intervals = [4, 8, 16, 32, 64, 128];
replot = true;

p_analytical = zeros(eval_num_samples, length(eval_num_intervals));
p_discrete   = zeros(eval_num_samples, length(eval_num_intervals));

e_d_true_list        = cell(1, length(eval_num_intervals));
lower_bound_e_d_list = cell(1, length(eval_num_intervals));
upper_bound_e_d_list = cell(1, length(eval_num_intervals));

if replot == false
    for isample = 1:eval_num_samples
        if mod(isample, 10) == 0
            disp(['Sample: ', num2str(isample)])
        end
        parameters = sample_parameters(1);
        parfor jj = 1:length(eval_num_intervals)
            M = eval_num_intervals(jj);
            x_full  = linspace(0, 1, M+1);
            x_inner = x_full(2:end-1);
            %% Analytical Solution
            u_analytical                = comp_analytical_state(x_inner, parameters);
            %% Discrete Solution
            u_discrete                  = comp_discrete_sol(parameters, M);
            %% Discretization Error
            e_d_true            =  u_analytical - u_discrete;
            e_d_true_list{jj}   = cat(1, e_d_true_list{jj}, e_d_true);
            %% Discretization Bound
            [lower_bound_e_d, upper_bound_e_d] = comp_discretization_error_bounds(parameters, M, 5);
            lower_bound_e_d_list{jj} = cat(1, lower_bound_e_d_list{jj}, lower_bound_e_d);
            upper_bound_e_d_list{jj} = cat(1, upper_bound_e_d_list{jj}, upper_bound_e_d);
            assert (sum(lower_bound_e_d < e_d_true) == M - 1, 'Invalid Lower bound')
            assert (sum(upper_bound_e_d > e_d_true) == M - 1, 'Invalid Upper bound')

            %% Figure for state and discretization error
            % if isample == 1 && jj == 1
            %     fg = figure();
            %     set(fg, 'Position', [100 1 600, 400]);
            %     [ha, pos] = tight_subplot(1,1,[0.1 .07],[.18 .1],[.1 .05]);
            %     axes(ha(1))
            %     y_lower = lower_bound_e_d; % Multiplied by 100 for visibility 
            %     y_upper = upper_bound_e_d;
            %     % l2 = plot(x_full, [0; e_d_true; 0], 'r', 'LineWidth', 2);
            %     plot(linspace(0, 1, 200), comp_analytical_state(linspace(0, 1, 200), parameters), 'Color', 'k', 'LineWidth', 2, 'DisplayName', '$u(x)$')
            %     hold on
            %     l2 = plot(x_inner, u_analytical, 'r', 'LineWidth', 2, 'DisplayName', '$u(x_i)$');
            %     l2.LineStyle = 'none'; l2.Marker = 'o'; l2.MarkerSize = 8; l2.MarkerFaceColor = 'r'; l2.MarkerEdgeColor = 'r';
            %     l1 = errorbar(x_inner, u_discrete, y_lower, y_upper, 's', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'Color', 'k', 'LineWidth', 2, 'DisplayName', '$\tilde{u}(x_i)$');
            %     l1.LineWidth = 2; l1.Color = 'k';l1.LineStyle='none';
            %     xlim([0.25, 0.75])
            %     hold off
            %     lg = legend();
            %     set(lg, 'Interpreter', 'latex', 'FontSize', 20, 'Location', 'northeast')
            %     title(['M = ', num2str(M)], 'Interpreter', 'latex', 'FontSize', 20)
            %     set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
            %     grid on; box on
            %     xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 20)
            %     xticks([0.25, 0.5, 0.75])
            %     % savefig('discretization_error_bounds_illustration.fig')
            % end

        end
    end
save('discretization_error_bounds_illustration.mat', 'e_d_true_list', 'lower_bound_e_d_list', 'upper_bound_e_d_list', 'eval_num_intervals')
end
data = load('discretization_error_bounds_illustration.mat');
e_d_true_list = data.e_d_true_list;
lower_bound_e_d_list = data.lower_bound_e_d_list;
upper_bound_e_d_list = data.upper_bound_e_d_list;
eval_num_intervals = data.eval_num_intervals;
plot_discretization_cdf(e_d_true_list, lower_bound_e_d_list, upper_bound_e_d_list, eval_num_intervals)

function t_true = comp_t_true(parameters, M, u_analytical)
    A   = get_A_mat(parameters, M);
    rhs = get_rhs(parameters, M);
    t_true = A * u_analytical - rhs;
end

function [t_sup, t_inf] = comp_t_bounds(parameters, M, lagrange_approx_order)
    % parameters: Parameters
    % M: Number of intervals
    % lagrange_approx_order: this order is approximated in the taylor series in the Lagrange remainder theorem
    dx = 1/M;
    x_full  = linspace(0, 1, M+1);
    x_inner = x_full(2:end-1);

    % Diagonals of the matrix A
    [sub_diag, main_diag, super_diag] = get_A_diags(parameters, M);

    [rhs, rhs_scalar] = get_rhs(parameters, M);

    t_sup = zeros(M - 1, 1);
    t_inf = zeros(M - 1, 1);

    num_lagrange_pts = 100000;
    if lagrange_approx_order == 3
        for ii = 1:length(x_inner)
            % Coefficients
            alpha_i = sub_diag(ii);
            beta_i  = main_diag(ii);
            nu_i    = super_diag(ii);

            % Lagrange parameters
            ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
            ci_plus  = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);

            % State and Derivatives at xi
            ui          = comp_analytical_state(x_inner(ii), parameters);
            ui_prime    = compute_state_derivative(x_inner(ii), parameters, 1);
            ui_dprime   = compute_state_derivative(x_inner(ii), parameters, 2);

            % Lagrange Remainder
            ui_tprime_ci_minus = compute_state_derivative(ci_minus, parameters, 3);
            ui_tprime_ci_plus  = compute_state_derivative(ci_plus, parameters, 3);

            closure_sup = (dx^3 / 6) * max(-alpha_i*ui_tprime_ci_minus + nu_i*ui_tprime_ci_plus);
            closure_inf = (dx^3 / 6) * min(-alpha_i*ui_tprime_ci_minus + nu_i*ui_tprime_ci_plus);

            % Gamma_i
            gamma_i = (alpha_i + beta_i + nu_i)*ui;
            gamma_i = gamma_i + (-alpha_i + nu_i)*dx*ui_prime;
            gamma_i = gamma_i + (alpha_i + nu_i)*(dx^2/2)*ui_dprime;
            gamma_i = gamma_i - rhs_scalar;

            % t bounds
            t_sup(ii) = gamma_i + closure_sup;
            t_inf(ii) = gamma_i + closure_inf;

        end
    elseif lagrange_approx_order == 4
        for ii = 1:length(x_inner)
            % Coefficients
            alpha_i = sub_diag(ii);
            beta_i  = main_diag(ii);
            nu_i    = super_diag(ii);

            % Lagrange parameters
            ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
            ci_plus  = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);

            % State and Derivatives at xi
            ui          = comp_analytical_state(x_inner(ii), parameters);
            ui_prime    = compute_state_derivative(x_inner(ii), parameters, 1);
            ui_dprime   = compute_state_derivative(x_inner(ii), parameters, 2);
            ui_tprime   = compute_state_derivative(x_inner(ii), parameters, 3);

            % Lagrange Remainder
            ui_4prime_ci_minus = compute_state_derivative(ci_minus, parameters, 4);
            ui_4prime_ci_plus  = compute_state_derivative(ci_plus, parameters, 4);

            closure_sup = (dx^4 / 24) * max(alpha_i*ui_4prime_ci_minus + nu_i*ui_4prime_ci_plus);
            closure_inf = (dx^4 / 24) * min(alpha_i*ui_4prime_ci_minus + nu_i*ui_4prime_ci_plus);

            % Gamma_i
            gamma_i = (alpha_i + beta_i + nu_i)*ui;
            gamma_i = gamma_i + (-alpha_i + nu_i)*dx*ui_prime;
            gamma_i = gamma_i + (alpha_i + nu_i)*(dx^2/2)*ui_dprime;
            gamma_i = gamma_i + (-alpha_i + nu_i)*(dx^3/6)*ui_tprime;
            gamma_i = gamma_i - rhs_scalar;

            % t bounds
            t_sup(ii) = gamma_i + closure_sup;
            t_inf(ii) = gamma_i + closure_inf;

        end

    elseif lagrange_approx_order == 5
        for ii = 1:length(x_inner)
            % Coefficients
            alpha_i = sub_diag(ii);
            beta_i  = main_diag(ii);
            nu_i    = super_diag(ii);

            % Lagrange parameters
            ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
            ci_plus  = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);

            % State and Derivatives at xi
            ui          = comp_analytical_state(x_inner(ii), parameters);
            ui_prime    = compute_state_derivative(x_inner(ii), parameters, 1);
            ui_dprime   = compute_state_derivative(x_inner(ii), parameters, 2);
            ui_tprime   = compute_state_derivative(x_inner(ii), parameters, 3);
            ui_4prime   = compute_state_derivative(x_inner(ii), parameters, 4);

            % Lagrange Remainder
            ui_5prime_ci_minus = compute_state_derivative(ci_minus, parameters, 5);
            ui_5prime_ci_plus  = compute_state_derivative(ci_plus, parameters, 5);

            closure_sup = (dx^5 / 120) * max(-alpha_i*ui_5prime_ci_minus + nu_i*ui_5prime_ci_plus);
            closure_inf = (dx^5 / 120) * min(-alpha_i*ui_5prime_ci_minus + nu_i*ui_5prime_ci_plus);

            % Gamma_i
            gamma_i = (alpha_i + beta_i + nu_i)*ui;
            gamma_i = gamma_i + (-alpha_i + nu_i)*dx*ui_prime;
            gamma_i = gamma_i + (alpha_i + nu_i)*(dx^2/2)*ui_dprime;
            gamma_i = gamma_i + (-alpha_i + nu_i)*(dx^3/6)*ui_tprime;
            gamma_i = gamma_i + (alpha_i + nu_i)*(dx^4/24)*ui_4prime;
            gamma_i = gamma_i - rhs_scalar;

            % t bounds
            t_sup(ii) = gamma_i + closure_sup;
            t_inf(ii) = gamma_i + closure_inf;

        end

    else
        error('Lagrange Approximation Order not supported')
    end

end


function u = comp_discrete_sol(parameters, M)
    A = get_A_mat(parameters, M);
    rhs = get_rhs(parameters, M);
    u = A\rhs;
    assert (size(u, 1) == M - 1, 'Discrete Solution has wrong dimensions')
end


function [error, error_norm] = comp_discretization_error(sol_analytical, sol_hp)
    error = sol_analytical - sol_hp; 
    if nargout > 1
        error_norm = norm(error, inf);
    end
end

function derivative = compute_state_derivative(x, parameters, order)
    % x: Input
    % parameters: Parameters
    % order: Order of the derivative

    %% Parameters Extraction
    [theta_1, theta_2] = get_parameters(parameters);

    if order == 1
        numerator = (50*theta_2^2).*(-(theta_1 ./ (1 + theta_1.*x)) + log(1 + theta_1));
        denominator = theta_1*log(1 + theta_1);
        derivative = -numerator./denominator;
    elseif order == 2
        numerator = 50*theta_1*theta_2^2;
        denominator = ((1 + theta_1.*x).^2) * log(1 + theta_1);
        derivative = -numerator./denominator;
    elseif order == 3
        numerator = 100*theta_1^2*theta_2^2;
        denominator = ((1 + theta_1.*x).^3) * log(1 + theta_1);
        derivative = numerator./denominator;
    elseif order == 4
        numerator = 300*theta_1^2*theta_2^2;
        denominator = ((1 + theta_1.*x).^4) * log(1 + theta_1);
        derivative = -numerator./denominator;
    elseif order == 5
        numerator = 1200*theta_1^3*theta_2^2;
        denominator = ((1 + theta_1.*x).^5) * log(1 + theta_1);
        derivative = numerator./denominator;
    else
        error('Order not supported')
    end
end

function [theta_1, theta_2] = get_parameters(parameters)
    theta_1 = parameters(1);
    theta_2 = parameters(2);
end

function [sub_diag, main_diag, super_diag] = get_A_diags(parameters, M)
    dx      = 1/M;
    idx     = 1:M;
    z       = idx * dx;

    [theta_1, theta_2] = get_parameters(parameters);
  
    constant_vector = 1 + theta_1 * (z - 0.5*dx);

    % Sub Diagonal
    sub_diag        = constant_vector(1:M - 1);

    % Super Diagonal
    super_diag      = constant_vector(2:M);

    % Main Diagonal
    main_diag       = -1 * (sub_diag + super_diag);
end

function A = get_A_mat(parameters, M)
    [sub_diag, main_diag, super_diag] = get_A_diags(parameters, M);

    A = diag(sub_diag(2:end), -1) + diag(main_diag, 0) + diag(super_diag(1:end-1), 1);

    assert (size(A, 1) == M-1 && size(A, 2) == M-1, 'Matrix A has wrong dimensions')

end


function [rhs, rhs_scalar] = get_rhs(parameters, M)
    dx = 1 / M;

    [theta_1, theta_2] = get_parameters(parameters);
    % Convert precision
    rhs     = -50*theta_2*theta_2* ones(M - 1, 1) * dx^2;
    assert (size(rhs, 1) == M - 1, 'RHS has wrong dimensions')
    if nargout > 1
        rhs_scalar = -50*theta_2*theta_2* dx^2;
    end
end

function [lower_bound, upper_bound] = comp_discretization_error_bounds(parameters, M, lagrange_approx_order)
    [t_inf, t_sup] = test2_bounds_t(parameters, M, lagrange_approx_order);
    t_inf = reshape(t_inf, [M-1, 1]);
    t_sup = reshape(t_sup, [M-1, 1]);   
    A = get_A_mat(parameters, M);
    lower_bound = A \ t_sup;
    upper_bound = A \ t_inf;
    assert (sum(lower_bound < upper_bound) == M - 1, 'Lower Bound is greater than Upper Bound')
end

function comp_state_lagrange_bounds(parameters, M, order)
    % parameters: Parameters
    % M: Number of intervals
    % lagrange_approx_order: this order is approximated in the taylor series in the Lagrange remainder theorem
    dx = 1/M;
    x_full  = linspace(0, 1, M+1);
    x_inner = x_full(2:end-1);

    ui_minus_lagrange_max = zeros(M-1, 1);
    ui_minus_lagrange_min = zeros(M-1, 1);
    ui_minus_true = zeros(M-1, 1);

    ui_plus_lagrange_max = zeros(M-1, 1);
    ui_plus_lagrange_min = zeros(M-1, 1);
    ui_plus_true = zeros(M-1, 1);

    num_lagrange_pts = 10000;
    for ii = 1:length(x_inner)
        ui_minus_true(ii) = comp_analytical_state(x_inner(ii) - dx, parameters);
        ui_plus_true(ii)  = comp_analytical_state(x_inner(ii) + dx, parameters);

        % State and Derivatives at xi
        ui          = comp_analytical_state(x_inner(ii), parameters);
        ui_prime    = compute_state_derivative(x_inner(ii), parameters, 1);
        ui_dprime   = compute_state_derivative(x_inner(ii), parameters, 2);

        P_minus = ui - dx*ui_prime + (dx^2/2)*ui_dprime;
        P_plus  = ui + dx*ui_prime + (dx^2/2)*ui_dprime;

        ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
        ci_plus  = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);

        ui_t_prime_ci_minus = compute_state_derivative(ci_minus, parameters, 3);
        ui_t_prime_ci_plus  = compute_state_derivative(ci_plus, parameters, 3);

        R_max_ci_minus = (dx^3 / 6) * max(-ui_t_prime_ci_minus);
        R_min_ci_minus = (dx^3 / 6) * min(-ui_t_prime_ci_minus);

        R_max_ci_plus = (dx^3 / 6) * max(ui_t_prime_ci_plus);
        R_min_ci_plus = (dx^3 / 6) * min(ui_t_prime_ci_plus);

        ui_minus_lagrange_max(ii) = P_minus + R_max_ci_minus;
        ui_minus_lagrange_min(ii) = P_minus + R_min_ci_minus;

        ui_plus_lagrange_max(ii)  = P_plus + R_max_ci_plus;
        ui_plus_lagrange_min(ii)  = P_plus + R_min_ci_plus;
    end
    fg = figure();
    fg.Position = [100, 100, 800, 400];
    subplot(1, 2, 1)
    plot(x_full(1:end-2), ui_minus_true, 'Color', 'k', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineStyle', 'none', 'DisplayName', '$u(x_i^-)$')
    hold on
    plot(x_full(1:end-2), ui_minus_lagrange_max, 'Color', 'r', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineStyle', '-', 'DisplayName', '$u(x_i^-) + R_{max}^-$')
    plot(x_full(1:end-2), ui_minus_lagrange_min, 'Color', 'b', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 5, 'MarkerFaceColor', 'b', 'LineStyle', '-', 'DisplayName', '$u(x_i^-) + R_{min}^-$')
    hold off
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
    lg = legend(); set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'best')
    subplot(1, 2, 2)
    plot(x_full(3:end), ui_plus_true, 'Color', 'k', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineStyle', 'none', 'DisplayName', '$u(x_i^+)$')
    hold on
    plot(x_full(3:end), ui_plus_lagrange_max, 'Color', 'r', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineStyle', '-', 'DisplayName', '$u(x_i^+) + R_{max}^+$')
    plot(x_full(3:end), ui_plus_lagrange_min, 'Color', 'b', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 5, 'MarkerFaceColor', 'b', 'LineStyle', '-', 'DisplayName', '$u(x_i^+) + R_{min}^+$')
    hold off
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
    lg = legend(); set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'best')


end

function [t_inf, t_sup]  = test1_bounds_t(parameters, M, order)
    %% Entire Objective is Maximized
    dx = 1/M;
    x_full  = linspace(0, 1, M+1);
    x_inner = x_full(2:end-1);

    [sub_diag, main_diag, super_diag] = get_A_diags(parameters, M);
    [rhs, rhs_scalar] = get_rhs(parameters, M);

    num_lagrange_pts = 10000;
    for ii = 1:length(x_inner)
        ui_true(ii)       = comp_analytical_state(x_inner(ii), parameters);
        ui_minus_true(ii) = comp_analytical_state(x_inner(ii) - dx, parameters);
        ui_plus_true(ii)  = comp_analytical_state(x_inner(ii) + dx, parameters);

        alpha_i = sub_diag(ii);
        beta_i  = main_diag(ii);
        nu_i    = super_diag(ii);
        
        t_true(ii) = alpha_i * ui_minus_true(ii) + beta_i * ui_true(ii) + nu_i * ui_plus_true(ii);
        t_true(ii) = t_true(ii) - rhs_scalar;

        %% Computes State and Derivatives
        ui = comp_analytical_state(x_inner(ii), parameters);
        ui_prime = compute_state_derivative(x_inner(ii), parameters, 1);
        ui_dprime = compute_state_derivative(x_inner(ii), parameters, 2);

        %% Lagrage approximation to ui_minus
        P =  ui - dx*ui_prime + (dx^2/2)*ui_dprime;
        ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
        R = (dx^3 / 6) * compute_state_derivative(ci_minus, parameters, 3);
        ui_minus_lag = P - R;

        %% Lagrage approximation to ui_plus
        P = ui + dx*ui_prime + (dx^2/2)*ui_dprime;
        ci_plus = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);
        R = (dx^3 / 6) * compute_state_derivative(ci_plus, parameters, 3);
        ui_plus_lag = P + R;

        t_approx = alpha_i * ui_minus_lag + beta_i * ui + nu_i * ui_plus_lag;
        t_approx = t_approx - rhs_scalar;

        t_sup(ii)    = max(t_approx);
        t_inf(ii)    = min(t_approx);

    end

end

function [t_inf, t_sup] = test2_bounds_t(parameters, M, order)
    %% First state u_{\pm 1} is bounded and then the rest is maximized
    dx = 1/M;
    x_full  = linspace(0, 1, M+1);
    x_inner = x_full(2:end-1);

    [sub_diag, main_diag, super_diag] = get_A_diags(parameters, M);
    [rhs, rhs_scalar] = get_rhs(parameters, M);

    num_lagrange_pts = 10000;
    for ii = 1:length(x_inner)
        ui_true(ii)       = comp_analytical_state(x_inner(ii), parameters);
        ui_minus_true(ii) = comp_analytical_state(x_inner(ii) - dx, parameters);
        ui_plus_true(ii)  = comp_analytical_state(x_inner(ii) + dx, parameters);

        alpha_i = sub_diag(ii);
        beta_i  = main_diag(ii);
        nu_i    = super_diag(ii);
        
        t_true(ii) = alpha_i * ui_minus_true(ii) + beta_i * ui_true(ii) + nu_i * ui_plus_true(ii);
        t_true(ii) = t_true(ii) - rhs_scalar;


        %% Computes State and Derivatives
        ui          = comp_analytical_state(x_inner(ii), parameters);
        ui_prime    = compute_state_derivative(x_inner(ii), parameters, 1);
        ui_dprime   = compute_state_derivative(x_inner(ii), parameters, 2);

        P_minus = ui - dx*ui_prime + (dx^2/2)*ui_dprime;
        P_plus  = ui + dx*ui_prime + (dx^2/2)*ui_dprime;

        ci_minus = linspace(x_inner(ii) - dx, x_inner(ii), num_lagrange_pts);
        ci_plus  = linspace(x_inner(ii), x_inner(ii) + dx, num_lagrange_pts);

        ui_t_prime_ci_minus = compute_state_derivative(ci_minus, parameters, 3);
        ui_t_prime_ci_plus  = compute_state_derivative(ci_plus, parameters, 3);

        R_max_ci_minus  = (dx^3 / 6) * max(-ui_t_prime_ci_minus);
        R_min_ci_minus  = (dx^3 / 6) * min(-ui_t_prime_ci_minus);

        R_max_ci_plus   = (dx^3 / 6) * max(ui_t_prime_ci_plus);
        R_min_ci_plus   = (dx^3 / 6) * min(ui_t_prime_ci_plus);

        %% Bounds for u(x_i^-)
        ui_minus_lagrange_max(ii) = P_minus + R_max_ci_minus;
        ui_minus_lagrange_min(ii) = P_minus + R_min_ci_minus;

        %% Bounds for u(x_i^+)
        ui_plus_lagrange_max(ii)  = P_plus + R_max_ci_plus;
        ui_plus_lagrange_min(ii)  = P_plus + R_min_ci_plus;

        %% Bounds for t(x_i)
        ui_minus_bounds = [ui_minus_lagrange_min(ii), ui_minus_lagrange_max(ii)];
        ui_plus_bounds  = [ui_plus_lagrange_min(ii), ui_plus_lagrange_max(ii)];

        %% T bounds
        t_sup(ii) = max(alpha_i*ui_minus_bounds + beta_i*ui + nu_i*ui_plus_bounds - rhs_scalar);
        t_inf(ii) = min(alpha_i*ui_minus_bounds + beta_i*ui + nu_i*ui_plus_bounds - rhs_scalar);

    end

end