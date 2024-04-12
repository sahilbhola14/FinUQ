function fwd_bound = comp_fwd_bound(vector_one, vector_two, bwd_bound)
    vector_one_dp   = convert_precision(vector_one, 'double');
    vector_two_dp   = convert_precision(vector_two, 'double');
    bwd_bound       = convert_precision(bwd_bound, 'double');

    numerator       = abs(vector_one_dp)*abs(vector_two_dp)';
    denominator     = abs(vector_one_dp*vector_two_dp');
    ratio           = numerator / denominator;
    fwd_bound       = bwd_bound.*ratio;
end