function [samples, perturbation] = sample_rel_error(n, precision)
    % Function samples Uniform[-unit_round_off, unit_round_off] 
    % n: number of samples
    % precision: precision of the samples
    urd     = comp_urd(precision);
    samples = urd*2.0*(rand(1, n) - 0.5);

    assert (isa(samples, 'double'));
    assert (length(samples) == n);

    if nargout > 1
        perturbation = 1 + samples; 
    end
end