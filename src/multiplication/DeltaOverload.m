classdef DeltaOverload
    %Delta Overload class for finite precision analysis

    % In finite precision analysis the arithmetic model \hat{a} = a(1 + \delta) is
    % often used. Furhter, arithmetic operations such as addition, subtraction, etc.
    % are also modeled as \hat{a op b} = (a op b) (1 + \delta).
    % This class implements the above arithmetic model for a single operand.
    % For example, operation a*b is carried out as \tilde{a}*\tilde{b}*(1 + \delta),
    % where \tilde{a} and \tilde{b} are given as \tilde{a} = a(1 + \delta_a) and
    % \tilde{b} = b(1 + \delta_b) respectively.


    properties
        val
        config % Dictionary with keys: "analysis", "precision"
    end

    methods
        function obj = DeltaOverload(val, config, initialize)
            obj.check_config(config);
            assert(size(val, 2) == 1, "DeltaOverload must be a column vector")
            if initialize == true
                % disp("Not sliced")
                d_samples = sample_one_plus_delta(config, size(val, 1));
                obj.val = val.*d_samples; % Error in representation is accounted
            elseif initialize == false
                % disp("sliced")
                d_samples = ones(size(val, 1), 1);
                obj.val = val.*d_samples; % Error in representation is not accounted
            else
                error("Invalid initialization")
            end
            obj.config = config;
        end

        function obj = check_config(obj, config)
            assert (isKey(config, "analysis"), "Invalid config")
            assert (isKey(config, "precision"), "Invalid config")
            assert (config("analysis") == "deterministic" || config("analysis") == "probabilistic" || config("analysis") == "off", "Invalid analysis type")
            assert (config("precision") == "single" || config("precision") == "half" || config("precision") == "double", "Invalid precision")
            if config("precision") == "double"
                assert (config("analysis") == "off", "Double precision is not supported for analysis")
            end
        end

        function obj = plus(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val + other.val)*d_samples;
        end

        function obj = minus(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val - other.val)*d_samples;
        end

        function obj = uminus(obj)
            num_ops = size(obj.val, 1);
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = -obj.val*d_samples;
        end

        function obj = uplus(obj)
            num_ops = size(obj.val, 1);
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = obj.val*d_samples;
        end

        function obj = times(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val.*other.val)*d_samples;
        end

        function obj = mtimes(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val*other.val)*d_samples;
        end

        function obj = mrdivide(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val/other.val)*d_samples;
        end

        function obj = rdivide(obj, other)
            num_ops = max(size(obj.val, 1), size(other.val, 1));
            assert (num_ops == 1, "Only one operation is currently supported")
            d_samples = sample_one_plus_delta(obj.config, 1);
            obj.val = (obj.val./other.val)*d_samples;
        end

    end
end

function one_plus_delta = sample_one_plus_delta(config, num_samples)
    if config("analysis") == "deterministic"
        sampling_function = @sample_deterministic;
        one_plus_delta = sampling_function(config, num_samples);
    elseif config("analysis") == "probabilistic"
        sampling_function = @sample_probabilistic;
        one_plus_delta = sampling_function(config, num_samples);
    elseif config("analysis") == "off"
        one_plus_delta = ones(num_samples, 1);
    else
        error("Invalid analysis type")
    end
end

function samples = sample_deterministic(config, num_samples)
    [meps, urd] = get_machine_epsilon(config("precision"));
    uniform_samples = rand(num_samples, 1);
    s = ones(num_samples, 1);
    s(uniform_samples < 0.5) = -1;
    samples = 1 + s.*urd;
end

function samples = sample_probabilistic(config, num_samples)
    keys = { 'num_samples', 'distribution', 'precision' };
    vals = { num_samples, 'model', config("precision") };
    args = containers.Map(keys, vals);
    samples = 1 + sample_rounding_error(args);
end

function [meps, urd] = get_machine_epsilon(precision)
    if precision == "double"
        meps = eps('double');
        urd = meps/2;
    elseif precision == "single"
        meps = double(eps('single'));
        urd = meps/2;
    elseif precision == "half"
        meps = double(eps('half'));
        urd = meps/2;
    else
        error("Invalid precision")
    end
end
