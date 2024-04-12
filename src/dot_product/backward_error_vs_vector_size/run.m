% Performs Scalar Multiplication when the numbers are sampled in the lower space 
void;

% Begin User Input
num_samples             = 10000;
k                       = 1; % Vectors are sampled U[2^{k}, 2^{k+1}]
sample_space            = 'StdNormal'; % Valid Cases: 'UZeroOne', 'UMOnePOne', 'UPowerTwo', 'StdNormal'
eval_vector_sizes       = 10.^(1:1:6);
test_confidence_levels  = [0.9];
precision               = "single";
% End User Input

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

        % Backward Error
        e_bwd(ii, jj)           = compute_bwd_error(dot_product_dp, dot_product_lp, vector_one_dp, vector_two_dp);
    
    end

end


mean_e_bwd = mean(e_bwd, 2);
max_e_bwd  = max(e_bwd, [], 2);


% Save the data
save(strcat('data_', convertStringsToChars(precision), 'k_', num2str(k), '_', sample_space, '_.mat'), 'eval_vector_sizes', 'eval_vector_sizes', 'deterministic_bwd_bound', 'higham_bwd_bound', 'bernstein_bwd_bound', 'mean_e_bwd', 'max_e_bwd', 'test_confidence_levels', 'sample_space', 'k', 'precision', 'num_samples');

line_set_width = 2.75;
fg = figure();
set(fg, 'Position', [100 100 800 500]);
[ha, pos] = tight_subplot(1,1,[.01 .03],[.18 .1],[.08 .03]);
axes(ha(1));
colors = get(gca,'colororder');
l0 = plot(eval_vector_sizes, deterministic_bwd_bound, 'DisplayName', '$\gamma_n$'); 
l0.Color = 'b'; l0.LineWidth = line_set_width; l0.MarkerFaceColor = l0.Color; l0.MarkerSize = 15; l0.Marker = 'd'; l0.MarkerEdgeColor = 'k';
hold on
for ii = 1:length(test_confidence_levels)
    l1 = plot(eval_vector_sizes, higham_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^h(\alpha = ', num2str(test_confidence_levels(ii)), '))$']);
    l1.Color = 'g'; l1.LineStyle = '--'; l1.LineWidth = line_set_width; l1.MarkerFaceColor = l1.Color; l1.Marker = 'o'; l1.MarkerSize = 15; l1.MarkerEdgeColor = 'k';
    l2 = plot(eval_vector_sizes, bernstein_bwd_bound(:, ii), 'DisplayName', ['$\tilde{\gamma}_n(\lambda_D^b(\alpha = ', num2str(test_confidence_levels(ii)), '))$']);
    l2.Color = 'm'; l2.LineStyle = '-.'; l2.LineWidth = line_set_width; l2.MarkerFaceColor = l2.Color; l2.Marker = 's'; l2.MarkerSize = 15; l2.MarkerEdgeColor = 'k';
end
l3 = plot(eval_vector_sizes, mean_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{mean}$');
l3.Color = 'r'; l3.LineWidth = line_set_width; l3.MarkerFaceColor = l3.Color; l3.Marker = '^'; l3.MarkerSize = 15; l3.MarkerEdgeColor = 'k';
l4 = plot(eval_vector_sizes, max_e_bwd, 'DisplayName', '$\epsilon_{bwd}^{max}$');
l4.Color = 'k'; l4.LineWidth = line_set_width; l4.MarkerFaceColor = l4.Color; l4.Marker = 'v'; l4.MarkerSize = 15; l4.MarkerEdgeColor = 'k';
xlabel('Vector size, $n$', 'Interpreter', 'latex')
lg = legend();
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'northwest')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.75)
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
hold off
if sample_space == "UZeroOne"
    saveas(fg, ['bwd_error_UZeroOne_', convertStringsToChars(precision), '.png'])
elseif sample_space == "UMOnePOne"
    saveas(fg, ['bwd_error_UMOnePOne_', convertStringsToChars(precision), '.png'])
elseif sample_space == "UPowerTwo"
    saveas(fg, ['bwd_error_UPowerTwo_',num2str(k),'_',convertStringsToChars(precision), '.png'])
elseif sample_space == "StdNormal"
    saveas(fg, ['bwd_error_StdNormal_', convertStringsToChars(precision), '.png'])
else
    error("Invalid Case");
end


%% Functions
function samples = get_vector(n, precision, k)
    samples = rand(1, n)*(2^k) + (2^k); % U[2^k, 2^{k+1}]
    samples = convert_precision(samples, precision);
end

function samples = get_vector_Stdnorm(n, precision)
    samples = randn(1, n);  % Standard Normal
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

function e_bwd = compute_bwd_error(dot_product_dp, dot_product_lp, vector_one_dp, vector_two_dp)

    dot_product_dp  = convert_precision(dot_product_dp, 'double');
    dot_product_lp  = convert_precision(dot_product_lp, 'double');
    vector_one_dp   = convert_precision(vector_one_dp, 'double');
    vector_two_dp   = convert_precision(vector_two_dp, 'double');

    numerator       = abs(dot_product_dp - dot_product_lp);
    denominator     = abs(vector_one_dp)*abs(vector_two_dp)';
    e_bwd           = numerator / denominator;
end