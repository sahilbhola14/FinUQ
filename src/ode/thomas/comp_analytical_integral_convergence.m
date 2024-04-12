compute_anal_solution_convergence();

%% Functions
function compute_anal_solution_convergence()
    num_param_samples   = 100000000;  % Number of parameter samples

    parameters          = sample_parameters(num_param_samples);

    anal_p              = compute_anal_p(parameters);

    step                = 100000;
    num_steps           = num_param_samples / step;
    mean_anal_p         = zeros(num_steps, 1);

    parfor ii =1:num_steps
        fprintf("Iteration %d of %d\n", ii, num_steps);
        sub_anal_p = anal_p(1:ii*step);
        mean_anal_p(ii) = mean(sub_anal_p);
    end

    input_samples       = (1:num_steps)*step;
    mean_anal_p         = mean_anal_p';

    save("anal_p_convergence.mat", "input_samples", "mean_anal_p");

    figure()
    l1 = plot((1:num_steps)*step, mean_anal_p);
    l1.LineWidth = 3;
    xlabel("Number of Samples, $n$", 'Interpreter', 'latex', 'FontSize', 20)
    ylabel("$\frac{1}{n}\sum_{i=1}^{n}p_{a}(\theta_1^i, \theta_2^i)$", 'Interpreter', 'latex', 'FontSize', 20)
    set(gca, 'XScale', 'log')
    title("Convergence of Analytical solution to the ODE", 'Interpreter', 'latex', 'FontSize', 20)
    set(gca, "FontSize", 20)
    grid on
    box on

end

function anal_p = compute_anal_p(parameters)
    theta_1 = convert_precision(parameters(:, 1), "double");
    theta_2 = convert_precision(parameters(:, 2), "double");
    numerator = (25.0*theta_2.^2).*(-2*theta_1 + (2 + theta_1).*log(1 + theta_1));
    denominator = (theta_1.^2).*log(1 + theta_1);
    anal_p = numerator./denominator;
end

