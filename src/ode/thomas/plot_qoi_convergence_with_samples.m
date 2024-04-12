function plot_qoi_convergence_with_samples(qoi_analytical_mean, qoi_hp_mean, qoi_lp_mean, qoi_model_mean, qoi_true_sol, eval_num_intervals, higher_precision, lower_precision, eval_parameter_samples, sample_mean_array, save_folder)
    if higher_precision == 'double'
        higher_precision_label = 'fp64';
    elseif higher_precision == 'single'
        higher_precision_label = 'fp32';
    else
        error('Invalid higher precision')
    end
    if lower_precision == 'single'
        lower_precision_label = 'fp32';
    elseif lower_precision == 'half'
        lower_precision_label = 'fp16';
    else
        error('Invalid lower precision')
    end
    fg = figure();
    set(fg, 'Position', [100 100 1500 500]);
    [ha, pos] = tight_subplot(1,3,[.01 .05],[.18 .09],[.1 .015]);

    plot_idx = length(eval_num_intervals) - 3 + 1;
    idx = 0;
    for ii = 1:3 
        idx = plot_idx + ii - 1;
        axes(ha(ii));
        l1 = plot(sample_mean_array, qoi_true_sol - qoi_analytical_mean(:,idx));
        l1.Color = 'r'; l1.LineWidth = 2;
        hold on
        l2 = plot(sample_mean_array, qoi_true_sol - qoi_hp_mean(:,idx));
        l2.Color = 'b'; l2.LineWidth = 2; l2.LineStyle = '-'; l2.Marker = 's'; l2.MarkerSize = 15; l2.MarkerFaceColor = l2.Color; l2.MarkerEdgeColor = 'k';

        l3 = plot(sample_mean_array, qoi_true_sol - qoi_lp_mean(:,idx));
        l3.Color = 'g'; l3.LineWidth = 2; l3.LineStyle = '-.';
        hold off

        title('$m$ = ' + string(eval_num_intervals(idx)), 'Interpreter', 'latex')
        set(gca, 'XScale', 'log')
        set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex')
        grid on; box on;

    end
end