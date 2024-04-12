void;
%% User Input
eval_num_intervals              = 2.^(2:7);                         % Number of intervals for the discretization of the spatial domain (ODE)
step_for_mean_convergence       = 1000;                              % Step for the mean convergence (Mean of solution is computed every 'step_for_mean_convergence' steps)
%% End User Input


% Single Precision
lower_precision                 = "single";                           % Lower precision for the computation (Used to enforce no representation error)
num_experiments                 = 100;                              % Number of experiments      
global save_folder
save_folder = lower_precision;

converged_mean_anal_p           = load("anal_p_convergence.mat");   % Obtained from the script "comp_analytical_integral_convergence.m"
qoi_true_sol = converged_mean_anal_p.mean_anal_p(end);

for ii =1:num_experiments
    fprintf("Experiment %d\n", ii)
    data = get_data("experiments/data_exp_" + string(ii));
    qoi_analytical                  = get_data_from_struc(data, "qoi_analytical");
    qoi_hp                          = get_data_from_struc(data, "qoi_hp");
    qoi_lp                          = get_data_from_struc(data, "qoi_lp");
    % Compute the mean
    for jj =1:length(eval_num_intervals)
        [abs_error_qoi_analytical_mean(ii, :, jj), sample_mean_array]  = comp_sample_mean(qoi_analytical(:, jj), step_for_mean_convergence);
        abs_error_qoi_analytical_mean(ii, :, jj)  = abs(qoi_true_sol - abs_error_qoi_analytical_mean(ii, :, jj));
        abs_error_qoi_hp_mean(ii, :, jj)          = comp_sample_mean(qoi_hp(:, jj), step_for_mean_convergence);
        abs_error_qoi_hp_mean(ii, :, jj)          = abs(qoi_true_sol - abs_error_qoi_hp_mean(ii, :, jj));
        abs_error_qoi_single_mean(ii, :, jj)          = comp_sample_mean(qoi_lp(:, jj), step_for_mean_convergence);
        abs_error_qoi_single_mean(ii, :, jj)          = abs(qoi_true_sol - abs_error_qoi_single_mean(ii, :, jj));
    end
end


% Half Precision
lower_precision                 = "half";                           % Lower precision for the computation (Used to enforce no representation error)
num_experiments                 = 58;                              % Number of experiments      
global save_folder
save_folder = lower_precision;

converged_mean_anal_p           = load("anal_p_convergence.mat");   % Obtained from the script "comp_analytical_integral_convergence.m"
qoi_true_sol = converged_mean_anal_p.mean_anal_p(end);

for ii =1:num_experiments
    fprintf("Experiment %d\n", ii)
    data = get_data("experiments/data_exp_" + string(ii));
    qoi_analytical                  = get_data_from_struc(data, "qoi_analytical");
    qoi_hp                          = get_data_from_struc(data, "qoi_hp");
    qoi_lp                          = get_data_from_struc(data, "qoi_lp");
    % Compute the mean
    for jj =1:length(eval_num_intervals)
        % [abs_error_qoi_analytical_mean(ii, :, jj), sample_mean_array]  = comp_sample_mean(qoi_analytical(:, jj), step_for_mean_convergence);
        % abs_error_qoi_analytical_mean(ii, :, jj)  = abs(qoi_true_sol - abs_error_qoi_analytical_mean(ii, :, jj));
        % abs_error_qoi_hp_mean(ii, :, jj)          = comp_sample_mean(qoi_hp(:, jj), step_for_mean_convergence);
        % abs_error_qoi_hp_mean(ii, :, jj)          = abs(qoi_true_sol - abs_error_qoi_hp_mean(ii, :, jj));
        abs_error_qoi_half_mean(ii, :, jj)          = comp_sample_mean(qoi_lp(:, jj), step_for_mean_convergence);
        abs_error_qoi_half_mean(ii, :, jj)          = abs(qoi_true_sol - abs_error_qoi_half_mean(ii, :, jj));
    end
end

% assert(1==2)

%% Compute the mean over experiments
mean_analytical = mean(abs_error_qoi_analytical_mean, 1);
std_analytical  = std(abs_error_qoi_analytical_mean, 1);

mean_hp = mean(abs_error_qoi_hp_mean, 1);
std_hp  = std(abs_error_qoi_hp_mean, 1);

mean_single = mean(abs_error_qoi_single_mean, 1);
std_single = std(abs_error_qoi_single_mean, 1);

mean_half = mean(abs_error_qoi_half_mean, 1);
std_half = std(abs_error_qoi_half_mean, 1);

fg = figure();
set(fg, 'Position', [100 1 1400 1200]);
[ha, pos] = tight_subplot(3,2,[0.1 .07],[.12 .05],[.09 .015]);
% y_lim_list = [0.8,]
for ii =1:length(eval_num_intervals)
    axes(ha(ii));
    analytical_data = [mean_analytical(1, 1, ii), mean_analytical(1, floor(end/2), ii), mean_analytical(1, end, ii)];
    hp_data = [mean_hp(1, 1, ii), mean_hp(1, floor(end/2), ii), mean_hp(1, end, ii)];
    single_data = [mean_single(1, 1, ii), mean_single(1, floor(end/2), ii), mean_single(1, end, ii)];
    half_data   = [mean_half(1, 1, ii), mean_half(1, floor(end/2), ii), mean_half(1, end, ii)];
    plot_data = [analytical_data; hp_data; single_data; half_data].';
    barlabels = {['$n=' + string(sample_mean_array(1)/1000) + '\times 10^3$'],...
                 ['$n=' + string(sample_mean_array(floor(end/2))/1000) + '\times 10^3$'],...
                 ['$n=' + string(sample_mean_array(end)/1000) + '\times 10^3$']
                };
    error_bar = [std_analytical(1, 1, ii), std_analytical(1, floor(end/2), ii), std_analytical(1, end, ii);...
                 std_hp(1, 1, ii), std_hp(1, floor(end/2), ii), std_hp(1, end, ii);...
                 std_single(1, 1, ii), std_single(1, floor(end/2), ii), std_single(1, end, ii)
                 std_half(1, 1, ii), std_half(1, floor(end/2), ii), std_half(1, end, ii)
                ];
    b = bar(plot_data, 'grouped');
    hold on
    for k = 1:numel(b)
        xtips = b(k).XEndPoints;
        ytips = b(k).YEndPoints;
        errorbar(xtips, ytips, error_bar(k, :), '.k', 'LineWidth', 1.5);
    end
    hold off
    xticklabels(barlabels);
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex', 'LineWidth', 1.5, 'ylim', [0, 1]);
    grid on; box on;
    title('$M$ = ' + string(eval_num_intervals(ii)), 'Interpreter', 'latex', 'FontSize', 20)
end

global_ylabel = '$\vert\mathrm{E}[P] - \frac{1}{n}\sum_i p_{i}^{\prime}\vert$';
annotation('textbox', [0.05, 0.4, 0.2, 0.05], 'String', global_ylabel, 'Interpreter', 'latex', 'FontSize', 20, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'Rotation', 90);

lg = legend('Analytical', 'fp64', 'fp32', 'fp16', 'Interpreter', 'latex');
set(lg, 'FontSize', 20, 'Interpreter', 'latex', 'Location', 'southoutside', 'NumColumns', 3);
lg.Position = [0.38, 0.01, 0.3, 0.05];

path        = strcat(save_folder, '/error_budget', '.fig');
savefig(fg, path);
saveas(fg, strrep(path, '.fig', '.png'));


%% Functions
function error = compute_abs_error(true_data, approx_data)
    error = abs(true_data - approx_data) * 100 / abs(true_data);
end

function data = get_data(filename)
    data = load(get_save_path(filename));
end

function path = get_save_path(file_name)
    save_folder = get_save_folder();
    path        = strcat(save_folder, "/", file_name, ".mat");
end

function folder = get_save_folder()
    global save_folder;
    folder = save_folder;
end

function update_save_folder(folder)
    global save_folder;
    save_folder = folder;
end

function data = get_data_from_struc(struc, field_name)
    if isfield(struc, field_name) == false
        error("Field '%s' not found in the structure", field_name);
    else
        data = struc.(field_name);
    end
end