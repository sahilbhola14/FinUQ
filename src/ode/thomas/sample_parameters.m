function parameters = sample_parameters(num_samples)
    % Sample parameters from the prior distribution
    % num_samples: number of samples to draw
    theta_1 = rand(num_samples, 1) + 0.1;
    theta_2 = rand(num_samples, 1) + 1.0;
    parameters = [theta_1, theta_2];

end