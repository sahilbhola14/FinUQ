void;

% Single Data
cdf_k_0_single = load('cdf_scalar_addition_k_0_num_samples_1000000_space_single.mat');
cdf_k_6_single = load('cdf_scalar_addition_k_6_num_samples_1000000_space_single.mat');


% Half Data
cdf_k_0_half = load('cdf_scalar_addition_k_0_num_samples_1000000_space_half.mat');
cdf_k_6_half = load('cdf_scalar_addition_k_6_num_samples_1000000_space_half.mat');

figure('Position', [10 10 1500 600]);
[ha, pos] = tight_subplot(1,2,[.01 .06],[.16 .08],[.06 .01]);

axes(ha(2)); 
    l1 = plot(cdf_k_0_single.X, cdf_k_0_single.F, "DisplayName", "True ($k=0$)"); 
    hold on
    l2 = plot(cdf_k_0_single.X_model, cdf_k_0_single.F_model, "DisplayName", "Model ($k=0$)");
    l1.LineWidth = 4.8; l2.LineWidth = 4.8;
    l1.Color = "k"; l2.Color = "r";
    l1.LineStyle = ":"; l2.LineStyle = ":";

    l3 = plot(cdf_k_6_single.X, cdf_k_6_single.F, "DisplayName", "True ($k=6$)");
    l4 = plot(cdf_k_6_single.X_model, cdf_k_6_single.F_model, "DisplayName", "Model ($k=6$)");
    l3.LineWidth = 3; l4.LineWidth = 3;
    l3.Color = "k"; l4.Color = "r";
    l3.LineStyle = "-"; l4.LineStyle = "-";

    xline(double(eps('single'))/2, "DisplayName", "$\mathrm{u}_{single}$", "LineWidth", 3, "Color", "b", "LineStyle", "--")
    grid on
    box on
    xlabel("$\xi_{+} = \vert \delta_{+}\vert$", "Interpreter", "latex", "FontSize", 20);
    ylabel("$F_{\Xi_{+}}(\xi_{+})$", "Interpreter", "latex", "FontSize", 20);
    title("fp32", "Interpreter", "latex", "FontSize", 20);

    lg = legend();
    set(lg, "Interpreter", "latex", "Location", "southeast", "FontSize", 20);
    set(gca, "FontSize", 20, "TickLabelInterpreter", "latex", "LineWidth", 1.75);
hold off

axes(ha(1)); 
    l1 = plot(cdf_k_0_half.X, cdf_k_0_half.F, "DisplayName", "True ($k=0$)"); 
    hold on
    l2 = plot(cdf_k_0_half.X_model, cdf_k_0_half.F_model, "DisplayName", "Model ($k=0$)");
    l1.LineWidth = 4.8; l2.LineWidth = 4.8;
    l1.Color = "k"; l2.Color = "r";
    l1.LineStyle = ":"; l2.LineStyle = ":";

    l3 = plot(cdf_k_6_half.X, cdf_k_6_half.F, "DisplayName", "True ($k=6$)");
    l4 = plot(cdf_k_6_half.X_model, cdf_k_6_half.F_model, "DisplayName", "Model ($k=6$)");
    l3.LineWidth = 3; l4.LineWidth = 3;
    l3.Color = "k"; l4.Color = "r";
    l3.LineStyle = "-"; l4.LineStyle = "-";
    xline(double(eps('half'))/2, "DisplayName", "$\mathrm{u}_{half}$", "LineWidth", 3, "Color", "b", "LineStyle", "--")
    grid on
    box on
    xlabel("$\xi_{+} = \vert \delta_{+}\vert$", "Interpreter", "latex", "FontSize", 20);
    ylabel("$F_{\Xi_{+}}(\xi_{+})$", "Interpreter", "latex", "FontSize", 20);
    title("fp16", "Interpreter", "latex", "FontSize", 20);

    lg = legend();
    set(lg, "Interpreter", "latex", "Location", "southeast", "FontSize", 20);
    set(gca, "FontSize", 20, "TickLabelInterpreter", "latex", "LineWidth", 1.75);
hold off
saveas(gcf,"cdf_scalar_addition.png")
