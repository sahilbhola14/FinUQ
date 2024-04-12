function samples = sample_rounding_error(args)
    %%%%%% %%%%% %%%%% %%%%% %%%%% %%%%%
    % args: Arguments for the test
    % Example of args:
        % keys = { 'num_samples', 'distribution', 'precision' };
        % vals = { 1000000, 'model', 'half' };
        % args = containers.Map(keys, vals);
    %%%%%% %%%%% %%%%% %%%%% %%%%% %%%%%

    % Check the Arguments
    check_args(args);
    % Samples
    if args('distribution') == "uniform"
        sampler = @get_uniform_samples;
    elseif args('distribution') == "normal"
        sampler = @get_normal_samples;
    elseif args('distribution') == "model"
        sampler = @get_model_samples;
    else
        error('Invalid distribution')
    end

    samples = sampler(args('num_samples'), args('precision'));
end

%% Additional Functions
function check_args(args)
    % Key checks
    assert(isKey(args, 'num_samples'), 'num_samples is required');
    assert(isKey(args, 'distribution'), 'distribution is required');
    assert(isKey(args, 'precision'), 'precision is required');

    % % Val checks
    valid_dists = {'uniform', 'normal', 'model'};
    valid_precisions = {'single', 'double', 'half'};

    assert(isnumeric(args('num_samples')) && args('num_samples') > 0, 'num_samples must be numeric AND > 0');
    assert(ismember(args('distribution'), valid_dists), 'distribution must be one of: uniform, normal, model');
    assert(ismember(args('precision'), valid_precisions), 'precision must be one of: single, double, half');
end

function [meps, urd] = get_machine_info(precision)
    if precision == "double"
        meps = eps('double');
        urd = eps('double')/2;
    elseif precision == "single"
        meps = double(eps('single'));
        urd = meps/2;
    elseif precision == "half"
        meps = double(eps('half'));
        urd = meps/2;
    else
        error('Invalid precision')
    end
end

function con_val = convert_precision(input_val, precision)
    % Function to convert the precision
    if precision == "double"
        con_val = double(input_val);
    elseif precision == "single"
        con_val = single(input_val);
    elseif precision == "half"
        con_val = half(input_val);
    else
        error('Invalid precision')
    end
end

function sample_uniform = get_uniform_samples(num_samples, precision)
    % Samples from a uniform distribution U(-u, u)
    [meps, urd] = get_machine_info(precision);
    sample_uniform = 2*urd*(rand(num_samples, 1) - 0.5);
    sample_uniform = convert_precision(sample_uniform, "double");

    assert (all(abs(sample_uniform) <= urd), 'Sample out of range')
end

function samples_model = get_model_samples(num_samples, precision)
    % Samples from the modeled distribution
    vector_one = get_standard_uniform_samples_minus_one_one(num_samples);
    vector_two = get_standard_uniform_samples_minus_one_one(num_samples);

    product_double = vector_one.*vector_two;
    product_conv = convert_precision(product_double, precision);

    samples_model = (product_conv - product_double) ./ product_double;

    samples_model = convert_precision(samples_model, "double");

    [meps, urd] = get_machine_info(precision);
    assert (all(abs(samples_model) <= urd), 'Sample out of range')
end

function samples_normal = get_normal_samples(num_samples, precision)
    % Samples from a normal distribution N(0, u)
    [meps, urd] = get_machine_info(precision);
    samples_normal = urd*randn(num_samples, 1);
    samples_normal = convert_precision(samples_normal, "double");
end

function samples_uniform = get_standard_uniform_samples_zero_one(num_samples)
    % Samples from a standard uniform distribution U(0, 1)
    samples_uniform = rand(num_samples, 1);
end

function samples_uniform = get_standard_uniform_samples_minus_one_one(num_samples)
    % Samples from a standard uniform distribution U(-1, 1)
    samples_uniform = 2*(rand(num_samples, 1) - 0.5);
end

function samples_normal = get_standard_normal_samples(num_samples)
    % Samples from a standard normal distribution N(0, 1)
    samples_normal = randn(num_samples, 1);
end

