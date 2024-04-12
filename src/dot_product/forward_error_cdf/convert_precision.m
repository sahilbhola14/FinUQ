function val_conv = convert_precision(val, precision)

    if precision == 'double'
        val_conv = double(val);
    elseif precision == 'single'
        val_conv = single(val);
    elseif precision == 'half'
        val_conv = half(val);
    else
        error('Invalid precision');
    end

end