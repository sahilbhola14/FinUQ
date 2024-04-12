function [samp_mean, num_samples] = comp_sample_mean(samples, step)
    % Function computes the sample mean 
    n         = length(samples);
    samples_d   = convert_precision(samples, 'double');
    assert (step > 0 && step <= n, 'Step size must be greater than 0 and less than the number of samples');
    num_steps = floor(n/step);
    for ii = 1:num_steps
        samp_mean(ii) = mean(samples_d(1:ii*step));
        num_samples(ii) = ii*step;
    end
    if mod(n, step) ~= 0
        samp_mean(num_steps+1) = mean(samples_d);
        num_samples(num_steps+1) = n;
    end

    assert (length(samp_mean) == num_steps + mod(n, step));

end