% Performs Scalar Multiplication when the numbers are sampled in the lower space 
void;

% Begin User Input
num_samples             = 100000;
k                       = 0; % Vectors are sampled U[2^{k}, 2^{k+1}]
sample_space            = 'UMOnePOne'; % Valid Cases: 'UZeroOne', 'UMOnePOne', 'UPowerTwo', 'StdNormal'
eval_vector_sizes       = [10000];
test_confidence_levels  = [0.9];
precision               = "single";
% End User Input

assert(length(eval_vector_sizes) == 1, "Only one vector size can be evaluated at a time");
assert(length(test_confidence_levels) == 1, "Only for the plotting it is required (Modifiy plot to accomodate multiple confidence levels)");

% Backward Error Bound
deterministic_bwd_bound = zeros(length(eval_vector_sizes), 1);
higham_bwd_bound        = zeros(length(eval_vector_sizes), length(test_confidence_levels));
bernstein_bwd_bound     = zeros(length(eval_vector_sizes), length(test_confidence_levels));
e_bwd                   = zeros(length(eval_vector_sizes), num_samples);

for ii = 1:length(eval_vector_sizes)
    fprintf("Evaluating Vector Size: %d\n", eval_vector_sizes(ii));
    % Deterministic Bound
    deterministic_bwd_bound(ii)     = dot_product_bwd_bound_deterministic(eval_vector_sizes(ii), precision);
    % Higham Bound 
    higham_bwd_bound(ii, :)         = dot_product_bwd_bound_higham(eval_vector_sizes(ii), precision, test_confidence_levels);
    % Variance-informed bound
    bernstein_bwd_bound(ii, :)      = dot_product_bwd_bound_bernstein(eval_vector_sizes(ii), precision, test_confidence_levels);

    % True Backward Error
    parfor jj = 1:num_samples
        fprintf("Percentage Complete: %.2f\n", (jj/num_samples)*100);
        % Vector (In the given precision)
        switch sample_space
            case 'UZeroOne'
                vector_one              = get_vector_U_zero_one(eval_vector_sizes(ii), precision);
                vector_two              = get_vector_U_zero_one(eval_vector_sizes(ii), precision);
            case 'UMOnePOne'
                vector_one              = get_vector_U_m_one_p_one(eval_vector_sizes(ii), precision);
                vector_two              = get_vector_U_m_one_p_one(eval_vector_sizes(ii), precision);
            case 'UPowerTwo'
                vector_one              = get_vector(eval_vector_sizes(ii), precision, k);
                vector_two              = get_vector(eval_vector_sizes(ii), precision, k);
            case 'StdNormal'
                vector_one              = get_vector_Stdnorm(eval_vector_sizes(ii), precision);
                vector_two              = get_vector_Stdnorm(eval_vector_sizes(ii), precision);
            otherwise
                error("Invalid Case");
        end

        % Vector (In double precision)
        vector_one_dp           = convert_precision(vector_one, 'double');
        vector_two_dp           = convert_precision(vector_two, 'double');

        % True Dot Product
        dot_product_dp          = compute_dot_product(vector_one_dp, vector_two_dp, "double");

        % Lower Precision Dot Product
        dot_product_lp          = compute_dot_product(vector_one, vector_two, precision);

        % Model Dot Product
        dot_product_model       = compute_model_dot_product(vector_one_dp, vector_two_dp, precision);

        % Backward Error
        e_bwd(ii, jj)           = compute_bwd_error(dot_product_dp, dot_product_lp, vector_one_dp, vector_two_dp);

        % Forward Error
        abs_error_true(ii, jj)  = comp_abs_error(dot_product_dp, dot_product_lp);
        abs_error_model(ii, jj) = comp_abs_error(dot_product_dp, dot_product_model);

        rel_error_true(ii, jj)  = comp_rel_error(dot_product_dp, dot_product_lp);
        rel_error_model(ii, jj) = comp_rel_error(dot_product_dp, dot_product_model);

        % Forward Error Bound
        deterministic_fwd_bound(ii, jj) = comp_fwd_bound(vector_one, vector_two, deterministic_bwd_bound(ii));
        higham_fwd_bound(ii, jj, :)     = comp_fwd_bound(vector_one, vector_two, higham_bwd_bound(ii, :));
        bernstein_fwd_bound(ii, jj, :)  = comp_fwd_bound(vector_one, vector_two, bernstein_bwd_bound(ii, :));
    
    end

end

% Figure
fg = figure();
set(fg, 'Position', [100 100 800 500]);
[ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .1],[.08 .03]);
axes(ha(1));

[rel_error_true_step_mean, step_vector]             = comp_sample_mean(rel_error_true(1, :), 1000);
[rel_error_model_step_mean, step_vector]            = comp_sample_mean(rel_error_model(1, :), 1000);
[deterministic_fwd_bound_step_mean, step_vector]    = comp_sample_mean(deterministic_fwd_bound(1, :), 1000);
[higham_fwd_bound_step_mean, step_vector]           = comp_sample_mean(higham_fwd_bound(1, :, 1), 1000);
[bernstein_fwd_bound_step_mean, step_vector]        = comp_sample_mean(bernstein_fwd_bound(1, :, 1), 1000);

l0 = plot(step_vector, rel_error_true_step_mean, 'DisplayName', '$\epsilon_{fwd}^{true}$');
% l0.Color = 'r'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.Marker = '^'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';
l0.Color = 'r'; l0.LineWidth = 2; l0.MarkerFaceColor = l0.Color; l0.Marker = 'none'; l0.MarkerSize = 15; l0.MarkerEdgeColor = 'k';

hold on
l1 = plot(step_vector, rel_error_model_step_mean, 'DisplayName', '$\epsilon_{fwd}^{model}$');
% l1.Color = 'k'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'v'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
l1.Color = 'k'; l1.LineWidth = 2; l1.MarkerFaceColor = l1.Color; l1.Marker = 'none'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';

l2 = plot(step_vector, deterministic_fwd_bound_step_mean, 'DisplayName', '$\mathcal{C}_{D}\gamma_n$');
% l2.Color = 'b'; l2.LineWidth = 2; l2.MarkerFaceColor = l2.Color; l2.Marker = 'd'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
l2.Color = 'b'; l2.LineWidth = 2; l2.MarkerFaceColor = l2.Color; l2.Marker = 'none'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';

l3 = plot(step_vector, higham_fwd_bound_step_mean, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = 2; l3.MarkerFaceColor = l3.Color; l3.Marker = 'o'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l3.Color = 'g'; l3.LineStyle = '--'; l3.LineWidth = 2; l3.MarkerFaceColor = l3.Color; l3.Marker = 'none'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';

l4 = plot(step_vector, bernstein_fwd_bound_step_mean, 'DisplayName', ['$\mathcal{C}_D\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(test_confidence_levels(1)), '))$']);
% l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 's'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';
l4.Color = 'm'; l4.LineStyle = '-.'; l4.LineWidth = 2; l4.MarkerFaceColor = l4.Color; l4.Marker = 'none'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';

if sample_space == "UZeroOne"
    title('Random uniform $\mathcal{U}[0, 1]$ vectors', 'Interpreter', 'latex', 'FontSize', 20)
elseif sample_space == "UMOnePOne"
    title('Random uniform $\mathcal{U}[-1, 1]$ vectors', 'Interpreter', 'latex', 'FontSize', 20)
elseif sample_space == "UPowerTwo"
    title(['Random uniform $\mathcal{U}[2^{', num2str(k), '}, 2^{', num2str(k+1), '}]$ vectors'], 'Interpreter', 'latex', 'FontSize', 20)
elseif sample_space == "StdNormal"
    title('Standard Normal vectors', 'Interpreter', 'latex', 'FontSize', 20)
else
    error("Invalid Case");
end

grid on; box on;
xlabel('$\#$ of dot product samples', 'Interpreter', 'latex')
lg = legend();
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
% ylim([-inf, 1e-5])

if sample_space == "UZeroOne"
    saveas(fg, ['fwd_error_vs_samples_UZeroOne_', convertStringsToChars(precision), '_vector_size_', num2str(eval_vector_sizes(1)), '.png'])
elseif sample_space == "UMOnePOne"
    saveas(fg, ['fwd_error_vs_samples_UMOnePOne_', convertStringsToChars(precision), '_vector_size_', num2str(eval_vector_sizes(1)), '.png'])
elseif sample_space == "UPowerTwo"
    saveas(fg, ['fwd_error_vs_samples_UPowerTwo_',num2str(k),'_',convertStringsToChars(precision), '_vector_size_', num2str(eval_vector_sizes(1)), '.png'])
elseif sample_space == "StdNormal"
    saveas(fg, ['fwd_error_vs_samples_StdNormal_', convertStringsToChars(precision), '_vector_size_', num2str(eval_vector_sizes(1)), '.png'])
else
    error("Invalid Case");
end


%% Functions
function samples = get_vector_Stdnorm(n, precision)
    samples = randn(1, n);  % Standard Normal
    samples = convert_precision(samples, precision);
end

function samples = get_vector(n, precision, k)
    samples = rand(1, n)*(2^k) + (2^k); % U[2^k, 2^{k+1}]
    samples = convert_precision(samples, precision);
end

function samples = get_vector_U_m_one_p_one(n, precision)
    samples = 2* (rand(1, n) - 0.5);  % U[-1, 1]
    samples = convert_precision(samples, precision);
end

function samples = get_vector_U_zero_one(n, precision)
    samples =  rand(1, n)          ;  % U[0, 1]
    samples = convert_precision(samples, precision);
end

function dot_product = compute_dot_product(vector_one, vector_two, precision)
    assert (length(vector_one) == length(vector_two));
    assert (isa(vector_one, precision) && isa(vector_two, precision));
    n = length(vector_one);
    summation = convert_precision(0, precision);
    for ii = 1:n
        summation = summation + vector_one(ii)*vector_two(ii);
    end
    dot_product = summation;
end

function dot_product = compute_model_dot_product(vector_one, vector_two, precision)
    assert (length(vector_one) == length(vector_two));
    assert (isa(vector_one, "double") && isa(vector_two, "double"));
    n = length(vector_one);
    summation = convert_precision(0, "double");
    for ii = 1:n
        [rel_error_samp , perturbation] = sample_rel_error(2, precision);
        summation = (summation + vector_one(ii)*vector_two(ii)*perturbation(1))*perturbation(2);
    end
    dot_product = summation;

    assert (isa(dot_product, "double"));
end

function e_bwd = compute_bwd_error(dot_product_dp, dot_product_lp, vector_one_dp, vector_two_dp)

    dot_product_dp  = convert_precision(dot_product_dp, 'double');
    dot_product_lp  = convert_precision(dot_product_lp, 'double');
    vector_one_dp   = convert_precision(vector_one_dp, 'double');
    vector_two_dp   = convert_precision(vector_two_dp, 'double');

    numerator       = abs(dot_product_dp - dot_product_lp);
    denominator     = abs(vector_one_dp)*abs(vector_two_dp)';
    e_bwd           = numerator / denominator;
end